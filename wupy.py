import wgpu

from wgpu.utils.compute import compute_with_buffers

device = wgpu.utils.get_default_device()


class ndarray:
    def __init__(self, shape, dtype="float32", *, _buf=None):
        assert _buf is not None
        self._buf = _buf
        self._shape = tuple(shape)
        self._dtype = str(dtype)
        assert self.ndim == 2, "arrays must be 2D for now"

    @property
    def ndim(self):
        return len(self._shape)

    @property
    def shape(self):
        return self._shape

    @property
    def dtype(self):
        return self._dtype

    def __repr__(self):
        shape_s = "x".join(str(x) for x in self.shape)
        dtype_s = self.dtype
        return f"<wupy.ndarray {shape_s} {dtype_s} >"

    def sum(self, axis=None):

        wgsl = """
            @group(0) @binding(0)
            var<storage,read> data_i1: array<f32>;

            @group(0) @binding(1)
            var<storage,read_write> data_o1: array<f32>;

            @compute
            @workgroup_size(1)
            fn main(@builtin(global_invocation_id) index3: vec3<u32>) {

                // info about the array
                let shape = array<i32,NDIM>SHAPE;
                let steps = array<i32,NDIM>STEPS;

                // Info about this invocation
                let index = i32(index3.x);

                var total = f32(0.0);
                LOOP
                data_o1[index] = total;
            }
        """

        shape = self.shape

        steps = shape[1:] + (1,)
        wgsl = (
            wgsl.replace("NDIM", str(self.ndim))
            .replace("SHAPE", str(shape))
            .replace("STEPS", str(steps))
        )

        calc_axis = axis
        if axis is None:
            if shape[0] >= 65535:
                calc_axis = 0
            elif shape[1] >= 65535:
                calc_axis = 1
            elif shape[0] < shape[1]:
                calc_axis = 0
            else:
                calc_axis = 1

        if calc_axis == 1:
            loop = """
                for (var i=0; i < shape[1]; i+=1) {
                    total += data_i1[steps[0] * index + i * steps[1]];
                }
            """
        elif calc_axis == 0:
            loop = """
                for (var i=0; i < shape[0]; i+=1) {
                    total += data_i1[steps[0] * i + index * steps[1]];
                }
            """
        wgsl = wgsl.replace("LOOP", loop)

        res_size = (shape[0] if calc_axis == 1 else shape[1]) * 4
        buffer2 = device.create_buffer(
            size=res_size, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC
        )

        res = run_shader_on_data(wgsl, self._buf, buffer2, res_size)
        if axis is None:
            return res.sum()
        else:
            return res


def run_shader_on_data(wgsl, buffer1, buffer2, n):
    cshader = device.create_shader_module(code=wgsl)

    # Setup layout and bindings
    binding_layouts = [
        {
            "binding": 0,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.read_only_storage,
            },
        },
        {
            "binding": 1,
            "visibility": wgpu.ShaderStage.COMPUTE,
            "buffer": {
                "type": wgpu.BufferBindingType.storage,
            },
        },
    ]
    bindings = [
        {
            "binding": 0,
            "resource": {"buffer": buffer1, "offset": 0, "size": buffer1.size},
        },
        {
            "binding": 1,
            "resource": {"buffer": buffer2, "offset": 0, "size": buffer2.size},
        },
    ]

    # Put everything together
    bind_group_layout = device.create_bind_group_layout(entries=binding_layouts)
    pipeline_layout = device.create_pipeline_layout(
        bind_group_layouts=[bind_group_layout]
    )
    bind_group = device.create_bind_group(layout=bind_group_layout, entries=bindings)

    # Create and run the pipeline
    compute_pipeline = device.create_compute_pipeline(
        layout=pipeline_layout,
        compute={"module": cshader, "entry_point": "main"},
    )

    t0 = perf_counter()
    command_encoder = device.create_command_encoder()
    compute_pass = command_encoder.begin_compute_pass()
    compute_pass.set_pipeline(compute_pipeline)
    compute_pass.set_bind_group(0, bind_group)
    compute_pass.dispatch_workgroups(n, 1, 1)  # x y z
    compute_pass.end()
    device.queue.submit([command_encoder.finish()])
    print(perf_counter() - t0)

    # Read result
    out = device.queue.read_buffer(buffer2).cast("f")
    return np.frombuffer(out, dtype=np.float32)  # out.tolist()


def _prod(shape):
    n = 1
    for s in shape:
        n *= s
    return n


def array(a, dtype=None, copy=True, order="K", subok=False, ndim=0, *, blocking=False):
    order = order or "K"
    ndim = ndim or 0

    nbytes = _prod(a.shape) * a.dtype.itemsize

    usage = (
        wgpu.BufferUsage.COPY_DST | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.STORAGE
    )
    buf = device.create_buffer(
        label="wupy.ndarray", size=nbytes, usage=usage, mapped_at_creation=False
    )

    device.queue.write_buffer(buf, 0, a, 0, nbytes)
    device.queue.submit([])

    return ndarray(shape=a.shape, dtype=a.dtype, _buf=buf)


def asarray(a, dtype=None, order=None, *, blocking=False):
    return array(a, dtype, copy=False, order=order, blocking=blocking)


###
from time import perf_counter


import numpy as np

a = np.linspace(0, 1, 2000 * 10000).astype(np.float32).reshape(10000, -1)

b = array(a)


# print(a.sum(0))

t0 = perf_counter()
res_a = a.sum(1)
print("np sum(1)", perf_counter() - t0)

t0 = perf_counter()
res_b = b.sum(1)
print("wgpu sum(1)", perf_counter() - t0)  # includes compile-time :/

assert np.allclose(res_a, res_b, rtol=1e-2)

print(a.sum())

print(b.sum())
