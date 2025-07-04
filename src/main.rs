use std::io;
use std::sync::Arc;
use std::thread;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use getrandom::getrandom;

use wgpu::util::DeviceExt;

fn main() {
    println!("蒙特卡洛方法计算π");
    println!("请选择计算方式:");
    println!("1. CPU多线程计算");
    println!("2. GPU计算");
    
    let mut choice = String::new();
    io::stdin()
        .read_line(&mut choice)
        .expect("读取选择失败");
    
    let use_gpu = match choice.trim() {
        "2" => true,
        _ => false,
    };
    
    println!("请输入采样点数量：");
    
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("读取输入失败");
    
    let n: u128 = match input.trim().parse() {
        Ok(num) => {
            if num == 0 {
                println!("采样点数量必须大于0，使用默认值 1000000");
                1000000
            } else {
                num
            }
        }
        Err(_) => {
            println!("输入无效，使用默认值 1000000");
            1000000
        }
    };
    
    // 收集用户随机输入作为熵源
    println!("\n为了提高随机性，请随机敲击键盘几次（按回车结束）：");
    println!("提示：请随机按任意字符键，然后按回车确认");
    
    let mut entropy_input = String::new();
    io::stdin()
        .read_line(&mut entropy_input)
        .expect("读取熵输入失败");
    
    // 从用户输入和系统随机源生成高质量熵
    let user_entropy = generate_crypto_entropy(&entropy_input);
    println!("已收集到 {} 字节的用户熵，并结合系统密码学随机源", entropy_input.len());
    
    println!("开始计算...");
    let start_time = std::time::Instant::now();
    
    let pi = if use_gpu {
        println!("使用GPU计算");
        pollster::block_on(calculate_pi_gpu(n, user_entropy))
    } else {
        println!("使用CPU多线程计算");
        calculate_pi_monte_carlo_mt(n, user_entropy)
    };
    
    let duration = start_time.elapsed();
    
    println!("使用 {} 个采样点", n);
    println!("计算得到的π值: {:.20}", pi);
    println!("真实π值:       {:.20}", std::f64::consts::PI);
    println!("绝对误差:      {:.20}", (pi - std::f64::consts::PI).abs());
    let relative_error = (pi - std::f64::consts::PI).abs() / std::f64::consts::PI;
    println!("相对误差:      {:.6e} ({:.4}%)", relative_error, relative_error * 100.0);
    println!("计算耗时: {:.2?}", duration);
}

// 使用密码学级别的随机数生成器生成高质量熵
fn generate_crypto_entropy(user_input: &str) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};
    
    // 从系统获取密码学安全的随机字节
    let mut crypto_bytes = [0u8; 32];
    getrandom(&mut crypto_bytes).expect("获取系统随机数失败");
    
    let mut hasher = DefaultHasher::new();
    
    // 哈希用户输入
    user_input.hash(&mut hasher);
    
    // 哈希密码学随机字节
    crypto_bytes.hash(&mut hasher);
    
    // 添加当前时间作为额外熵
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    now.hash(&mut hasher);
    
    // 添加输入长度和字符统计
    user_input.len().hash(&mut hasher);
    
    // 对每个字符的位置和值进行哈希
    for (i, ch) in user_input.chars().enumerate() {
        (i, ch as u32).hash(&mut hasher);
    }
    
    // 额外混合密码学随机数
    let mut crypto_u64 = [0u8; 8];
    getrandom(&mut crypto_u64).expect("获取系统随机数失败");
    let extra_entropy = u64::from_le_bytes(crypto_u64);
    
    hasher.finish() ^ extra_entropy
}

fn calculate_pi_monte_carlo_mt(n: u128, user_entropy: u64) -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    // 获取系统CPU核心数
    let num_threads = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4) as u128;
    
    println!("使用 {} 个线程", num_threads);
    
    // 使用原子类型来安全地累计结果
    let inside_circle = Arc::new(AtomicU64::new(0));
    
    // 计算每个线程处理的采样点数
    let samples_per_thread = n / num_threads;
    let remainder = n % num_threads;
    
    let mut handles = Vec::new();
    
    for i in 0..num_threads {
        let inside_circle_clone = Arc::clone(&inside_circle);
        
        // 为最后一个线程分配剩余的采样点
        let samples = if i == num_threads - 1 {
            samples_per_thread + remainder
        } else {
            samples_per_thread
        };
        
        let handle = thread::spawn(move || {
            use std::time::{SystemTime, UNIX_EPOCH};
            
            // 每个线程使用密码学级别的随机数生成器
            let mut crypto_seed = [0u8; 32];
            getrandom(&mut crypto_seed).expect("获取系统随机数失败");
            
            // 将用户熵混合到种子中
            let user_entropy_bytes = user_entropy.to_le_bytes();
            let thread_entropy = (i as u64).to_le_bytes();
            
            // 混合多种熵源
            for j in 0..8 {
                crypto_seed[j] ^= user_entropy_bytes[j % 8];
                crypto_seed[j + 8] ^= thread_entropy[j % 8];
            }
            
            let mut rng = StdRng::from_seed(crypto_seed);
            let mut local_inside = 0;
            
            for _ in 0..samples {
                // 使用密码学级别随机数生成 [-1, 1] 范围内的随机点
                let x: f64 = rng.gen_range(-1.0..1.0);
                let y: f64 = rng.gen_range(-1.0..1.0);
                
                // 检查点是否在单位圆内
                if x * x + y * y <= 1.0 {
                    local_inside += 1;
                }
            }
            
            // 原子地累加到全局计数器
            inside_circle_clone.fetch_add(local_inside, Ordering::Relaxed);
        });
        
        handles.push(handle);
    }
    
    // 等待所有线程完成
    for handle in handles {
        handle.join().unwrap();
    }
    
    let total_inside = inside_circle.load(Ordering::Relaxed);
    
    // π = 4 * (圆内点数 / 总点数)
    4.0 * total_inside as f64 / n as f64
}

// GPU计算相关结构
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuInput {
    samples_per_workgroup: u32,
    extra_samples: u32,
    user_entropy_low: u32,
    user_entropy_high: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GpuOutput {
    inside_count: u32,
}

async fn calculate_pi_gpu(n: u128, user_entropy: u64) -> f64 {
    calculate_pi_gpu_impl(n, user_entropy).await
}

async fn calculate_pi_gpu_impl(n: u128, user_entropy: u64) -> f64 {
    // 初始化GPU设备
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::all(),
        ..Default::default()
    });
    
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        )
        .await
        .unwrap();

    println!("GPU设备: {}", adapter.get_info().name);
    
    // 获取GPU的真实限制
    let limits = device.limits();
    let max_compute_workgroups_per_dimension = limits.max_compute_workgroups_per_dimension;
    let max_buffer_size = limits.max_buffer_size;
    
    println!("GPU限制: 最大工作组数 = {}, 最大缓冲区大小 = {} MB", 
             max_compute_workgroups_per_dimension, 
             max_buffer_size / 1024 / 1024);
    
    // GPU分批处理策略
    let max_samples_per_workgroup = 200000u32; // 每组最多20万样本
    let max_samples_per_batch = max_compute_workgroups_per_dimension as u128 * max_samples_per_workgroup as u128;
    
    println!("GPU单批次最大处理能力: {} 个采样点", max_samples_per_batch);
    
    if n <= max_samples_per_batch {
        // 单批次处理
        println!("使用单批次GPU处理");
        calculate_pi_gpu_batch(&device, &queue, n, max_compute_workgroups_per_dimension, user_entropy).await
    } else {
        // 多批次处理
        let num_batches = ((n as f64) / (max_samples_per_batch as f64)).ceil() as u128;
        println!("数据量过大，将分 {} 个批次进行GPU处理", num_batches);
        
        let mut total_inside = 0u128;
        let mut processed_samples = 0u128;
        
        for batch in 0..num_batches {
            let batch_start = batch * max_samples_per_batch;
            let batch_end = std::cmp::min((batch + 1) * max_samples_per_batch, n);
            let batch_size = batch_end - batch_start;
            
            println!("处理批次 {}/{}: {} 个采样点", batch + 1, num_batches, batch_size);
            
            let batch_pi = calculate_pi_gpu_batch(&device, &queue, batch_size, max_compute_workgroups_per_dimension, user_entropy.wrapping_add(batch as u64)).await;
            let batch_inside = (batch_pi * batch_size as f64 / 4.0) as u128;
            
            total_inside += batch_inside;
            processed_samples += batch_size;
            
            // 显示进度
            let progress = ((batch + 1) as f64 / num_batches as f64) * 100.0;
            println!("批次 {}/{} 完成，进度: {:.1}%", batch + 1, num_batches, progress);
        }
        
        println!("GPU分批处理完成: 总圆内点数 = {}, 总样本数 = {}", total_inside, processed_samples);
        4.0 * total_inside as f64 / processed_samples as f64
    }
}

async fn calculate_pi_gpu_batch(device: &wgpu::Device, queue: &wgpu::Queue, n: u128, max_workgroups: u32, user_entropy: u64) -> f64 {
    let max_samples_per_workgroup = 200000u32;
    
    // 计算这个批次的工作组配置
    let ideal_workgroups = std::cmp::min(
        max_workgroups,
        ((n as f64) / (max_samples_per_workgroup as f64)).ceil() as u32
    );
    
    let actual_workgroups = if ideal_workgroups == 0 { 1 } else { ideal_workgroups };
    let samples_per_workgroup = (n / actual_workgroups as u128) as u32;
    let extra_samples = (n % actual_workgroups as u128) as u32;
    
    // 创建计算着色器
    let compute_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Monte Carlo Pi Compute Shader"),
        source: wgpu::ShaderSource::Wgsl(r#"
struct Input {
    samples_per_workgroup: u32,
    extra_samples: u32,
    user_entropy_low: u32,
    user_entropy_high: u32,
}

struct Output {
    inside_count: u32,
}

@group(0) @binding(0)
var<uniform> input: Input;

@group(0) @binding(1)
var<storage, read_write> output: array<Output>;

// 改进的随机数生成器 - 使用多种算法混合获得更好的随机性
fn wang_hash(seed: u32) -> u32 {
    var s = seed;
    s = (s ^ 61u) ^ (s >> 16u);
    s = s * 9u;
    s = s ^ (s >> 4u);
    s = s * 0x27d4eb2du;
    s = s ^ (s >> 15u);
    return s;
}

fn xorshift32(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x = x ^ (x << 13u);
    x = x ^ (x >> 17u);
    x = x ^ (x << 5u);
    *state = x;
    return x;
}

fn pcg_hash(input: u32) -> u32 {
    var state = input * 747796405u + 2891336453u;
    var word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn better_random(state: ptr<function, u32>, seed_offset: u32) -> u32 {
    // 结合多种随机数算法以获得更好的随机性
    var xor_result = xorshift32(state);
    var wang_result = wang_hash(*state + seed_offset);
    var pcg_result = pcg_hash(xor_result ^ wang_result);
    
    // 更新状态
    *state = xor_result ^ pcg_result;
    
    return pcg_result;
}

fn random_f32(state: ptr<function, u32>, seed_offset: u32) -> f32 {
    return f32(better_random(state, seed_offset)) / 4294967296.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let workgroup_id = global_id.x;
    if (workgroup_id >= arrayLength(&output)) {
        return;
    }
    
    // 使用更复杂的种子生成策略，结合用户熵
    var user_entropy = (input.user_entropy_high << 16u) | input.user_entropy_low;
    var base_seed = 12345u + workgroup_id * 1013904223u + user_entropy;
    var time_seed = workgroup_id * 982451653u + 1234567u + (user_entropy >> 16u);
    var rng_state = wang_hash(base_seed) ^ pcg_hash(time_seed);
    
    // 确保种子不为0
    if (rng_state == 0u) {
        rng_state = 1u;
    }
    
    var inside_count = 0u;
    
    // 基础样本数
    var samples_to_process = input.samples_per_workgroup;
    
    // 前extra_samples个工作组需要处理额外的一个样本
    if (workgroup_id < input.extra_samples) {
        samples_to_process = samples_to_process + 1u;
    }
    
    // 处理分配给这个工作组的所有样本
    for (var i = 0u; i < samples_to_process; i++) {
        // 为每个样本使用不同的种子偏移
        let seed_offset = i * 2654435761u + workgroup_id;
        let x = random_f32(&rng_state, seed_offset) * 2.0 - 1.0;
        let y = random_f32(&rng_state, seed_offset + 1u) * 2.0 - 1.0;
        
        let distance_squared = x * x + y * y;
        if (distance_squared <= 1.0) {
            inside_count = inside_count + 1u;
        }
    }
    
    // 将结果写入输出缓冲区
    output[workgroup_id].inside_count = inside_count;
}
        "#.into()),
    });

    // 创建输入缓冲区
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Input Buffer"),
        contents: bytemuck::cast_slice(&[GpuInput {
            samples_per_workgroup: samples_per_workgroup,
            extra_samples: extra_samples,
            user_entropy_low: (user_entropy & 0xFFFFFFFF) as u32,
            user_entropy_high: (user_entropy >> 32) as u32,
        }]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // 创建输出缓冲区
    let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output Buffer"),
        size: (actual_workgroups * std::mem::size_of::<GpuOutput>() as u32) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 创建读取缓冲区
    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Staging Buffer"),
        size: output_buffer.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    // 创建绑定组布局
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });

    // 创建绑定组
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output_buffer.as_entire_binding(),
            },
        ],
    });

    // 创建计算管线
    let compute_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Compute Pipeline Layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Compute Pipeline"),
        layout: Some(&compute_pipeline_layout),
        module: &compute_shader,
        entry_point: "main",
    });

    // 执行计算
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Encoder"),
    });

    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&compute_pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(actual_workgroups, 1, 1);
    }

    encoder.copy_buffer_to_buffer(&output_buffer, 0, &staging_buffer, 0, output_buffer.size());
    queue.submit(std::iter::once(encoder.finish()));

    // 读取结果
    let buffer_slice = staging_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |v| sender.send(v).unwrap());
    device.poll(wgpu::Maintain::Wait);

    if let Some(Ok(())) = receiver.receive().await {
        let data = buffer_slice.get_mapped_range();
        let result: &[GpuOutput] = bytemuck::cast_slice(&data);
        
        let total_inside: u64 = result.iter().map(|r| r.inside_count as u64).sum();
        
        drop(data);
        staging_buffer.unmap();
        
        // 返回这个批次的π估计值
        4.0 * total_inside as f64 / n as f64
    } else {
        panic!("GPU批次计算失败");
    }
}
