// 蒙特卡洛计算π的GPU着色器

struct Input {
    total_samples: u32,
    seed: u32,
}

struct Output {
    inside_count: u32,
}

@group(0) @binding(0)
var<uniform> input: Input;

@group(0) @binding(1)
var<storage, read_write> output: array<Output>;

// 简单的线性同余随机数生成器
fn random(state: ptr<function, u32>) -> f32 {
    *state = (*state * 1103515245u + 12345u);
    return f32(*state) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    
    // 每个工作项使用不同的随机种子
    var rng_state = input.seed + index * 1000000u;
    var inside_count = 0u;
    
    // 处理分配给这个工作组的样本
    for (var i = 0u; i < input.total_samples; i++) {
        // 生成 [-1, 1] 范围内的随机点
        let x = random(&rng_state) * 2.0 - 1.0;
        let y = random(&rng_state) * 2.0 - 1.0;
        
        // 检查点是否在单位圆内
        if (x * x + y * y <= 1.0) {
            inside_count++;
        }
    }
    
    // 存储结果
    output[index].inside_count = inside_count;
}
