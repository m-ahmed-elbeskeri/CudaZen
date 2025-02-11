import re

class CudaZenTranspiler:
    def __init__(self, dsl_code):
        """
        A DSL -> C++ CUDA transpiler that supports:
         - debug_mode(true|false)
         - set_device(N);
         - kernel/gpu_only/cpu_only/gpu_and_cpu function definitions
         - allocations: cpu/gpu/pinned/unified (optional manual_delete)
         - manual deallocation: deallocate(varName);
         - auto deallocation for non-manual_delete variables
         - async copies: copy_to_gpu_async(...), copy_to_cpu_async(...)
         - kernel launches
         - graph capturing with graph(name, stream), plus launch_graph(name)
         - if_in_bounds(...) checks (1D, 2D, 3D)
         - synchronize;
         - SCOPES: auto-deallocation now happens at the end of the same block { ... }.
        """
        self.dsl_lines = dsl_code.split('\n')
        self.output_lines = []
        self.debug_mode = None

        # variable allocations: var_name -> (var_type, alloc_type, num_elems)
        self.variables = {}

        # A stack of scopes; each scope has an 'auto_free' set for variable names.
        self.scope_stack = []

        # Variables that are manually deallocated.
        self.manual_deallocations = {}

        # Streams defined in the DSL.
        self.streams = set()

        # Graphs: name -> (stream_name, capturing_bool)
        self.graphs = {}

    def transpile(self):
        # Insert includes.
        self.emit('#include <cuda_runtime.h>')
        self.emit('#include <cstdlib>')
        self.emit('#include <cuda_runtime_api.h>')
        self.emit('#include <cuda_graphs.h>')
        self.emit('')

        # Push a "root" scope.
        self.push_scope()

        # Check for debug_mode in DSL.
        for line in self.dsl_lines:
            if line.strip().startswith("debug_mode("):
                mode = line.strip()[11:-2].lower()
                self.debug_mode = (mode == "true")

        # Define CHECK_CUDA macro based on debug_mode.
        if self.debug_mode:
            self.emit('#define CHECK_CUDA(call) do { \\')
            self.emit('    cudaError_t err = call; \\')
            self.emit('    if (err != cudaSuccess) { \\')
            self.emit('        fprintf(stderr, "CUDA error: %s\\n", cudaGetErrorString(err)); \\')
            self.emit('        exit(EXIT_FAILURE); \\')
            self.emit('    } \\')
            self.emit('} while(0)')
        else:
            self.emit('#define CHECK_CUDA(call) call')
        self.emit('')

        # Process each DSL line.
        for line in self.dsl_lines:
            processed_line = self.handle_line(line)
            if processed_line is not None:
                self.output_lines.append(processed_line)

        # Pop any remaining scopes.
        while len(self.scope_stack) > 1:
            self.pop_scope()
        if len(self.scope_stack) == 1:
            self.pop_scope()

        # Substitute global_thread_index calls with full expressions.
        self.substitute_thread_ids()

        return "\n".join(self.output_lines)

    def push_scope(self):
        """Push a new scope with an empty set of auto-free variable names."""
        self.scope_stack.append({'auto_free': set()})

    def pop_scope(self):
        """Pop the top scope, generating auto-deallocation lines for its variables."""
        scope = self.scope_stack.pop()
        auto_vars = scope['auto_free']
        if auto_vars:
            self.output_lines.append('// Auto-deallocate variables for this scope')
            for var_name in auto_vars:
                self.output_lines.append(self.generate_deallocation(var_name))

    def emit(self, text):
        self.output_lines.append(text)

    def handle_line(self, line):
        stripped = line.strip()

        # Skip debug_mode(...) lines.
        if stripped.startswith("debug_mode("):
            return ''

        # 1) Function definitions.
        patterns = {
            'kernel': '__global__',
            'gpu_only': '__device__',
            'cpu_only': '__host__',
            'gpu_and_cpu': '__host__ __device__'
        }
        for dsl_word, prefix in patterns.items():
            func_re = re.match(rf'{dsl_word}\s+(\w+)\(([^)]*)\)(\s*){{?', stripped)
            if func_re:
                func_name = func_re.group(1)
                func_args = func_re.group(2)
                line_has_brace = stripped.endswith("{")
                if line_has_brace:
                    self.push_scope()
                return f'{prefix} void {func_name}({func_args})' + (" {" if line_has_brace else "")

        # 2) set_device(N);
        set_dev = re.match(r'set_device\((\d+)\);', stripped)
        if set_dev:
            return f'CHECK_CUDA(cudaSetDevice({set_dev.group(1)}));'

        # 3) deallocate(varName);
        dealloc_re = re.match(r'deallocate\((\w+)\);', stripped)
        if dealloc_re:
            var_name = dealloc_re.group(1)
            self.manual_deallocations[var_name] = True
            for scope in reversed(self.scope_stack):
                if var_name in scope['auto_free']:
                    scope['auto_free'].remove(var_name)
                    break
            return self.generate_deallocation(var_name)

        # 4) Graph creation.
        graph_open = re.match(r'graph\((\w+),\s*(\w+)\)\s*{', stripped)
        if graph_open:
            gname, sname = graph_open.groups()
            self.graphs[gname] = (sname, True)
            if sname not in self.streams:
                self.streams.add(sname)
                self.emit(f'cudaStream_t {sname};\nCHECK_CUDA(cudaStreamCreate(&{sname}));')
            self.push_scope()
            return (
                f'cudaGraph_t {gname};\n'
                f'cudaGraphExec_t {gname}_instance;\n'
                f'CHECK_CUDA(cudaStreamBeginCapture({sname}, cudaStreamCaptureModeGlobal));'
            )

        # 5) End of block.
        if stripped == '}':
            active_graph = None
            for gname, (sname, capturing) in self.graphs.items():
                if capturing:
                    active_graph = (gname, sname)
                    break
            self.pop_scope()
            if active_graph:
                gname, sname = active_graph
                self.graphs[gname] = (sname, False)
                return (f'}} // end scope\nCHECK_CUDA(cudaStreamEndCapture({sname}, &{gname}));\n'
                        f'CHECK_CUDA(cudaGraphInstantiate(&{gname}_instance, {gname}, NULL, NULL, 0));')
            else:
                return '}'

        # 6) launch_graph(gName);
        glaunch = re.match(r'launch_graph\((\w+)\);', stripped)
        if glaunch:
            gname = glaunch.group(1)
            if gname not in self.graphs:
                return f'// Graph {gname} not defined'
            return f'CHECK_CUDA(cudaGraphLaunch({gname}_instance, 0));'

        # 7) Async copies.
        copy_gpu_async = re.match(r'copy_to_gpu_async\((\w+),\s*(\w+),\s*(\w+)\)(?:\s*on\s+(\w+))?;', stripped)
        if copy_gpu_async:
            dev_ptr, host_ptr, num_elems, st = copy_gpu_async.groups()
            stream_part = '0'
            if st:
                if st not in self.streams:
                    self.streams.add(st)
                    self.emit(f'cudaStream_t {st};\nCHECK_CUDA(cudaStreamCreate(&{st}));')
                stream_part = st
            return (f'CHECK_CUDA(cudaMemcpyAsync({dev_ptr}, {host_ptr}, '
                    f'{num_elems} * sizeof(*{host_ptr}), cudaMemcpyHostToDevice, {stream_part}));')

        copy_cpu_async = re.match(r'copy_to_cpu_async\((\w+),\s*(\w+),\s*(\w+)\)(?:\s*on\s+(\w+))?;', stripped)
        if copy_cpu_async:
            host_ptr, dev_ptr, num_elems, st = copy_cpu_async.groups()
            stream_part = '0'
            if st:
                if st not in self.streams:
                    self.streams.add(st)
                    self.emit(f'cudaStream_t {st};\nCHECK_CUDA(cudaStreamCreate(&{st}));')
                stream_part = st
            return (f'CHECK_CUDA(cudaMemcpyAsync({host_ptr}, {dev_ptr}, '
                    f'{num_elems} * sizeof(*{host_ptr}), cudaMemcpyDeviceToHost, {stream_part}));')

        # 8) synchronize;
        if stripped == 'synchronize;':
            return 'CHECK_CUDA(cudaDeviceSynchronize());'

        # 9) if_in_bounds expansions.
        ib1 = re.match(r'if_in_bounds\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\)', stripped)
        if ib1:
            idx_var, lower, upper, stride = ib1.groups()
            return f'if (({idx_var} >= {lower}) && ({idx_var} < {upper})) {{'

        ib2 = re.match(r'if_in_bounds2D\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)', stripped)
        if ib2:
            varx, vary, sx, ex, sy, ey = ib2.groups()
            return (f'if (({varx} >= {sx}) && ({varx} < {ex}) && '
                    f'({vary} >= {sy}) && ({vary} < {ey})) {{')

        ib3 = re.match(
            r'if_in_bounds3D\(\s*([^,]+),\s*([^,]+),\s*([^,]+),'
            r'\s*([^,]+),\s*([^,]+),\s*([^,]+),'
            r'\s*([^,]+),\s*([^,]+),\s*([^)]+)\)',
            stripped
        )
        if ib3:
            varx, vary, varz, sx, ex, sy, ey, sz, ez = ib3.groups()
            return (f'if (({varx} >= {sx}) && ({varx} < {ex}) && '
                    f'({vary} >= {sy}) && ({vary} < {ey}) && '
                    f'({varz} >= {sz}) && ({varz} < {ez})) {{')

        # 10) Allocations.
        alloc_regex = (
            r'(cpu|gpu|pinned|unified)\s+'
            r'(\w+\*?)\s+'
            r'(\w+)\s*=\s*'
            r'alloc_\1<([^>]+)>\('
            r'([^,)]+)(?:,\s*([\w|]+))?\)'
            r'(?:\s*manual_delete)?;'
        )
        alloc_m = re.match(alloc_regex, stripped)
        if alloc_m:
            alloc_type, var_type, var_name, base_type, num_elems, options = alloc_m.groups()
            if not options:
                options = ''
            self.variables[var_name] = (var_type, alloc_type, num_elems)
            if 'manual_delete' not in stripped:
                if self.scope_stack:
                    self.scope_stack[-1]['auto_free'].add(var_name)
            if alloc_type == 'cpu':
                return f'{var_type} {var_name} = new {base_type}[{num_elems}];'
            elif alloc_type == 'gpu':
                return (
                    f'{var_type} {var_name};\n'
                    f'CHECK_CUDA(cudaMalloc((void**)&{var_name}, {num_elems} * sizeof({base_type})));'
                )
            elif alloc_type == 'unified':
                return (
                    f'{var_type} {var_name};\n'
                    f'CHECK_CUDA(cudaMallocManaged((void**)&{var_name}, {num_elems} * sizeof({base_type})));'
                )
            elif alloc_type == 'pinned':
                if not options:
                    options = 'cudaHostAllocDefault'
                return (
                    f'{var_type} {var_name};\n'
                    f'CHECK_CUDA(cudaHostAlloc((void**)&{var_name}, {num_elems} * sizeof({base_type}), {options}));'
                )

        # 11) Kernel launch.
        launch_regex = (
            r'launch\s+(\w+)\(([^)]*)\)\s+with\s+\{\s*threads:([^,]+),\s*blocks:([^,}]+)'
            r'(?:,\s*shared_mem:([^,}]+))?'
            r'(?:,\s*stream:\s*(\w+))?'
            r'\s*\};'
        )
        lmatch = re.match(launch_regex, stripped)
        if lmatch:
            kernel_name, args, threads, blocks, smem, sname = lmatch.groups()
            stream_part = ''
            if sname:
                if sname not in self.streams:
                    self.streams.add(sname)
                    self.emit(f'cudaStream_t {sname};\nCHECK_CUDA(cudaStreamCreate(&{sname}));')
                stream_part = f', {sname}'
            if smem:
                return f'{kernel_name}<<<{blocks}, {threads}, {smem}{stream_part}>>>({args});'
            else:
                return f'{kernel_name}<<<{blocks}, {threads}{stream_part}>>>({args});'

        # If the line ends with '{' but hasn't been handled, push a new scope.
        if stripped.endswith('{'):
            self.push_scope()
            return line

        # Otherwise, pass the line as-is.
        return line

    def generate_deallocation(self, var_name):
        """Generate code to deallocate var_name based on its allocation type."""
        if var_name in self.variables:
            _, alloc_type, _ = self.variables[var_name]
            if alloc_type == 'cpu':
                return f'delete[] {var_name};'
            elif alloc_type == 'pinned':
                return f'CHECK_CUDA(cudaFreeHost({var_name}));'
            elif alloc_type in ('gpu', 'unified'):
                return f'CHECK_CUDA(cudaFree({var_name}));'
        return f'// Cannot deallocate unknown variable {var_name}'

    def substitute_thread_ids(self):
        """
        Replace occurrences of global_thread_index(x), global_thread_index(y)
        or global_thread_index(z) with the full expressions.
        """
        for i, line in enumerate(self.output_lines):
            line = re.sub(r'\bglobal_thread_index\s*\(\s*x\s*\)',
                          'blockIdx.x * blockDim.x + threadIdx.x', line)
            line = re.sub(r'\bglobal_thread_index\s*\(\s*y\s*\)',
                          'blockIdx.y * blockDim.y + threadIdx.y', line)
            line = re.sub(r'\bglobal_thread_index\s*\(\s*z\s*\)',
                          'blockIdx.z * blockDim.z + threadIdx.z', line)
            self.output_lines[i] = line


if __name__ == "__main__":
    example_dsl = r"""
debug_mode(true);
set_device(0);

#N = 1000;

cpu_only vector_add_cpu(float* a, float* b, float* c, int n){
  for(int i = 0; i < n; i++){
    c[i] = a[i] + b[i];
  }
}

kernel vector_add_gpu(float* a, float* b, float* c, int n){
  // Using global_thread_index to compute the global thread index.
  int i = global_thread_index(x);
  if(i < n){
    c[i] = a[i] + b[i];
  }
}

int main() {
  cpu float* h_a = alloc_cpu<float>(N);
  cpu float* h_b = alloc_cpu<float>(N) manual_delete;
  pinned float* pinnedBuf = alloc_pinned<float>(256);

  {
    gpu float* d_a = alloc_gpu<float>(N);
    unify float* d_b = alloc_unified<float>(N); // Note: "unify" typo remains, so this line is passed as-is.
  }

  // Allocation outside of inner blocks.
  gpu float* d_c = alloc_gpu<float>(N);

  {
    // A small inner block.
    pinned float* pinned2 = alloc_pinned<float>(512);
  }

  launch vector_add_gpu(d_a, d_b, d_c, N) with { threads:256, blocks:10 };
  synchronize;

  return 0;
}
"""

    transpiler = CudaZenTranspiler(example_dsl)
    final_code = transpiler.transpile()
    print(final_code)
