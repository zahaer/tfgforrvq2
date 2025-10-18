#!/usr/bin/env python3
"""
环境设置脚本
用于设置RQ-VAE LoRA训练环境
"""

import os
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """运行命令并处理错误"""
    print(f"\n{'='*50}")
    print(f"执行: {description}")
    print(f"命令: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print("✅ 成功!")
        if result.stdout:
            print("输出:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 失败!")
        print("错误:", e.stderr)
        return False


def check_cuda():
    """检查CUDA环境"""
    print("\n🔍 检查CUDA环境...")
    
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
            print(f"   GPU数量: {torch.cuda.device_count()}")
            return True
        else:
            print("❌ CUDA不可用")
            return False
    except ImportError:
        print("❌ PyTorch未安装")
        return False


def install_requirements():
    """安装依赖包"""
    print("\n📦 安装依赖包...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt文件不存在")
        return False
    
    return run_command(
        f"pip install -r {requirements_file}",
        "安装Python依赖包"
    )


def setup_wandb():
    """设置Wandb"""
    print("\n🔧 设置Wandb...")
    
    wandb_key = input("请输入Wandb API Key (可选，直接回车跳过): ").strip()
    if wandb_key:
        return run_command(
            f"wandb login {wandb_key}",
            "登录Wandb"
        )
    else:
        print("⏭️  跳过Wandb设置")
        return True


def create_directories():
    """创建必要的目录"""
    print("\n📁 创建目录结构...")
    
    directories = [
        "lora_data",
        "lora_outputs", 
        "checkpoints",
        "logs",
        "results"
    ]
    
    base_dir = Path(__file__).parent
    
    for dir_name in directories:
        dir_path = base_dir / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"✅ 创建目录: {dir_path}")
    
    return True


def test_imports():
    """测试关键包的导入"""
    print("\n🧪 测试包导入...")
    
    packages = [
        ("torch", "PyTorch"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("accelerate", "Accelerate"),
        ("wandb", "Wandb"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas")
    ]
    
    failed_imports = []
    
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
        except ImportError as e:
            print(f"❌ {name}: {e}")
            failed_imports.append(name)
    
    if failed_imports:
        print(f"\n❌ 以下包导入失败: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ 所有包导入成功!")
        return True


def setup_huggingface():
    """设置Hugging Face"""
    print("\n🤗 设置Hugging Face...")
    
    hf_token = input("请输入Hugging Face Token (用于下载LLaMA2，可选): ").strip()
    if hf_token:
        return run_command(
            f"huggingface-cli login --token {hf_token}",
            "登录Hugging Face"
        )
    else:
        print("⏭️  跳过Hugging Face设置")
        print("💡 提示: 如果需要下载LLaMA2模型，请手动设置Hugging Face Token")
        return True


def main():
    """主函数"""
    print("🚀 RQ-VAE LoRA训练环境设置")
    print("="*50)
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ 需要Python 3.8或更高版本")
        sys.exit(1)
    
    print(f"✅ Python版本: {sys.version}")
    
    # 执行设置步骤
    steps = [
        ("创建目录", create_directories),
        ("安装依赖", install_requirements),
        ("检查CUDA", check_cuda),
        ("设置Hugging Face", setup_huggingface),
        ("设置Wandb", setup_wandb),
        ("测试导入", test_imports)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        if not step_func():
            failed_steps.append(step_name)
    
    # 总结
    print("\n" + "="*50)
    print("🎯 环境设置完成!")
    
    if failed_steps:
        print(f"❌ 以下步骤失败: {', '.join(failed_steps)}")
        print("请手动解决这些问题后重新运行")
    else:
        print("✅ 所有步骤都成功完成!")
        print("\n📋 下一步:")
        print("1. 准备RQ-VAE代码数据")
        print("2. 运行 dataset_generator.py 生成训练数据")
        print("3. 运行 train_lora.py 开始训练")
        print("4. 运行 inference.py 测试模型")
    
    print("="*50)


if __name__ == "__main__":
    main()
