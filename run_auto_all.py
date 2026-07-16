"""run_auto_all.py — 一键在服务器上跑完所有数据集 + SBM regimes 的自动 config 实验

功能
====
  1. 通过 deploy.py 部署代码到服务器
  2. 远程后台运行 `auto_config.py --all --seeds 3`
  3. 等待完成
  4. 拉回结果（auto_config_result.md + auto_config_decisions.jsonl + 日志）

等价命令（手动执行）：
  python deploy.py all --script auto_config.py --log logs/auto_config_run.log --args "--all --seeds 3"

本脚本只是封装，方便记忆和一键运行。

用法
====
  python run_auto_all.py              # 默认：3 seeds，完整实验
  python run_auto_all.py --seeds 5    # 5 seeds（更稳定但更慢）
  python run_auto_all.py --seeds 1    # 1 seed（快速验证，约 5 分钟）
  python run_auto_all.py --dry-run    # 只打印命令，不执行（检查用）
"""
import argparse
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(
        description="一键在服务器上跑 auto_config.py --all 完整实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--seeds", type=int, default=3,
                        help="随机种子数（默认 3；SBM regimes 用 5）")
    parser.add_argument("--epochs", type=int, default=None,
                        help="训练轮数（默认：auto_config.py 内部真实 200, SBM 300）")
    parser.add_argument("--output", default="results/auto_config_result.md",
                        help="结果报告文件名（默认 results/auto_config_result.md）")
    parser.add_argument("--log", default="logs/auto_config_run.log",
                        help="日志文件名（默认 logs/auto_config_run.log）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印要执行的命令，不执行（检查用）")
    parser.add_argument("--yes", "-y", action="store_true",
                        help="跳过确认提示，直接执行（用于后台/非交互运行）")
    parser.add_argument("--resume", action="store_true",
                        help="从断点恢复（跳过已完成的 task，仅配合 --all 使用）")
    parser.add_argument("--fresh", action="store_true",
                        help="忽略断点文件，从头开始（删除旧断点）")
    parser.add_argument("--clear-logs", action="store_true",
                        help="清空旧决策日志（避免 auto_config_decisions.jsonl 多次重跑后膨胀）")
    args = parser.parse_args()

    # 构造传给 auto_config.py 的参数
    script_args = f"--all --seeds {args.seeds} --output {args.output}"
    if args.epochs is not None:
        script_args += f" --epochs {args.epochs}"
    # 断点恢复 flag 透传
    if args.resume:
        script_args += " --resume"
    elif args.fresh:
        script_args += " --fresh"
    if args.clear_logs:
        script_args += " --clear-logs"

    # 构造 deploy.py 命令
    deploy_cmd = [
        sys.executable, "deploy.py", "all",
        "--script", "auto_config.py",
        "--log", args.log,
        "--args", script_args,
    ]

    print("=" * 60)
    print("一键自动 config 实验")
    print("=" * 60)
    print(f"  脚本      : auto_config.py")
    print(f"  脚本参数  : {script_args}")
    print(f"  日志文件  : {args.log}")
    print(f"  输出报告  : {args.output}")
    print(f"  deploy 命令: {' '.join(deploy_cmd)}")
    print("=" * 60)

    if args.dry_run:
        print("\n[dry-run] 仅打印命令，不执行。")
        return

    # 确认开始（--yes 跳过，用于后台/非交互运行）
    if not args.yes:
        print(f"\n即将在服务器上运行（setup + run + 等待 + fetch）。")
        print(f"预计耗时：3 seeds × (3 真实数据集 + 4 SBM regimes) ≈ 30-45 分钟")
        confirm = input("继续？[y/N] ").strip().lower()
        if confirm != 'y':
            print("已取消。")
            return

    # 执行 deploy.py all
    print(f"\n[{time.strftime('%H:%M:%S')}] 开始部署并运行...")
    result = subprocess.run(deploy_cmd)
    if result.returncode != 0:
        print(f"\n[错误] deploy.py 退出码 {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[{time.strftime('%H:%M:%S')}] 全部完成！")
    print(f"\n查看结果：")
    print(f"  - 结果报告  : {args.output}")
    print(f"  - 决策日志  : results/auto_config_decisions.jsonl")
    print(f"  - 运行日志  : {args.log}")


if __name__ == "__main__":
    main()
