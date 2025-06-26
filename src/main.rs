use std::io;
use std::sync::Arc;
use std::thread;

fn main() {
    println!("蒙特卡洛方法计算π");
    println!("请输入采样点数量：");
    
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("读取输入失败");
    
    let n: u64 = match input.trim().parse() {
        Ok(num) => {
            if num == 0 {
                println!("采样点数量必须大于0，使用默认值 1000000");
                1000000
            } else if num > 1_000_000_000_000 {
                println!("采样点数量太大，限制为最大值 1,000,000,000,000");
                1_000_000_000_000
            } else {
                num
            }
        }
        Err(_) => {
            println!("输入无效，使用默认值 1000000");
            1000000
        }
    };
    
    println!("开始计算，使用多线程...");
    let start_time = std::time::Instant::now();
    let pi = calculate_pi_monte_carlo_mt(n);
    let duration = start_time.elapsed();
    
    println!("使用 {} 个采样点", n);
    println!("计算得到的π值: {}", pi);
    println!("真实π值: {}", std::f64::consts::PI);
    println!("误差: {}", (pi - std::f64::consts::PI).abs());
    println!("计算耗时: {:.2?}", duration);
}

fn calculate_pi_monte_carlo_mt(n: u64) -> f64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    
    // 获取系统CPU核心数
    let num_threads = thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4) as u64;
    
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
            
            // 每个线程使用不同的随机种子
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64 + i * 1000000;
            
            let mut rng = SimpleRng::new(seed);
            let mut local_inside = 0;
            
            for _ in 0..samples {
                // 生成 [-1, 1] 范围内的随机点
                let x = rng.next_f64() * 2.0 - 1.0;
                let y = rng.next_f64() * 2.0 - 1.0;
                
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

// 简单的线性同余生成器
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed }
    }
    
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(1103515245).wrapping_add(12345);
        self.state
    }
    
    fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }
}
