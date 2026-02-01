#!/usr/bin/env python3
"""
模型推理测试脚本
用于验证模型加载和推理是否正常

使用方法:
    python test_inference.py [模型路径] [测试图片1] [测试图片2] ...
    
示例:
    python test_inference.py \\
        /path/to/checkpoint-8000-merged \\
        /path/to/test1.jpg \\
        /path/to/test2.jpg
"""

import sys
import time
from pathlib import Path
from inference_example import load_model, infer_single_image

# 默认配置 (可通过命令行覆盖)
DEFAULT_MODEL_PATH = "/scratch/li96/zl9731/cs16/Model/output/VLM/v5.4-3/checkpoint-8000-merged"
DEFAULT_TEST_IMAGES = [
    # 提供几张测试图片路径 (需要替换为实际路径)
    # "/path/to/test1.jpg",
    # "/path/to/test2.jpg",
]

# 解析命令行参数
if len(sys.argv) > 1:
    MODEL_PATH = sys.argv[1]
    TEST_IMAGES = sys.argv[2:] if len(sys.argv) > 2 else DEFAULT_TEST_IMAGES
else:
    MODEL_PATH = DEFAULT_MODEL_PATH
    TEST_IMAGES = DEFAULT_TEST_IMAGES

def test_model():
    """测试模型推理"""
    print("=" * 70)
    print("VLM 模型推理测试")
    print("=" * 70)
    
    # 1. 检查模型路径
    if not Path(MODEL_PATH).exists():
        print(f"❌ 模型路径不存在: {MODEL_PATH}")
        print(f"\n💡 提示: 请运行 merge_lora 脚本将 checkpoint 合并为完整模型")
        sys.exit(1)
    print(f"✅ 模型路径: {MODEL_PATH}")
    
    # 检查测试图片
    if not TEST_IMAGES:
        print(f"\n⚠️  未指定测试图片")
        print(f"\n使用方法:")
        print(f"  python test_inference.py [模型路径] [测试图片1] [测试图片2] ...")
        sys.exit(1)
    
    valid_images = [p for p in TEST_IMAGES if Path(p).exists()]
    if not valid_images:
        print(f"\n❌ 所有测试图片均不存在:")
        for p in TEST_IMAGES:
            print(f"  - {p}")
        sys.exit(1)
    
    print(f"✅ 测试图片数量: {len(valid_images)}")
    
    # 2. 加载模型
    print("\n" + "=" * 70)
    print("[1/2] 加载模型...")
    start = time.time()
    try:
        model, processor = load_model(
            MODEL_PATH, 
            device="cuda:0",
            use_4bit=False,  # 可改为 True 节省显存
            is_merged=True
        )
        load_time = time.time() - start
        print(f"✅ 模型加载成功 (耗时: {load_time:.2f}s)")
        
        # 打印模型信息
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   参数量: {total_params / 1e9:.2f}B")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # 3. 测试推理
    print("\n" + "=" * 70)
    print("[2/2] 推理测试...")
    
    total_time = 0
    success_count = 0
    
    for i, img_path in enumerate(valid_images, 1):
        print(f"\n--- 测试 {i}/{len(valid_images)} ---")
        print(f"图片: {img_path}")
        
        start = time.time()
        try:
            result = infer_single_image(
                model, 
                processor, 
                img_path,
                temperature=0.2,
                max_new_tokens=512
            )
            infer_time = time.time() - start
            total_time += infer_time
            success_count += 1
            
            print(f"✅ 推理成功 (耗时: {infer_time:.2f}s)")
            print(f"\n结果:")
            print("-" * 70)
            # 输出前500字符，避免过长
            if len(result) > 500:
                print(result[:500] + "\n... (结果已截断)")
            else:
                print(result)
            print("-" * 70)
            
        except Exception as e:
            print(f"❌ 推理失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 4. 统计信息
    print("\n" + "=" * 70)
    print("测试完成")
    print("=" * 70)
    print(f"成功: {success_count}/{len(valid_images)}")
    if success_count > 0:
        avg_time = total_time / success_count
        print(f"平均推理时间: {avg_time:.2f}s")
        print(f"吞吐量: {1/avg_time:.2f} 张/秒")
    print("=" * 70)

if __name__ == "__main__":
    test_model()
