"""deploy.py — 一键部署到服务器并运行实验（GPU 加速）。

服务器：remember@10.25.64.102（有 GPU，Python 环境已就绪）
本脚本在本地 Windows 运行，通过 SSH/SCP 远程操作。

用法：
  python deploy.py setup     # 传代码 + 数据到服务器（首次必跑）
  python deploy.py run       # 远程后台跑 main.py（nohup，SSH 断开不影响）
  python deploy.py status    # 查看远程实验是否在跑 + 当前进度
  python deploy.py tail      # 实时查看远程日志（Ctrl+C 退出）
  python deploy.py fetch     # 拉回 results/ + image/ + logs/ 到本地
  python deploy.py all       # setup + run + 轮询等完成 + fetch（一键到底）
  python deploy.py all_baseline  # 多卡并行跑 run_wj.py + 合并 + SOTA 对比

目录结构（本地和服务器一致）：
  - wj/        WJ 核心包
  - sota/      SOTA 对比工具包
  - results/   实验结果（JSON + markdown 报告）
  - logs/      实验日志
  - image/     图表输出

注意：
  - 服务器路径默认 ~/WJ，可在下方 REMOTE_DIR 修改
  - 首次 setup 后，代码改动只需再跑 setup 增量同步
  - run 是后台运行，SSH 断开实验继续跑；用 status/tail 监控
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ============================== 配置（按需修改） ==============================
SERVER = "remember@10.25.64.102"          # SSH 目标（用户@主机）
REMOTE_DIR = "~/WJ"                        # 服务器工作目录（用户已有此目录）
# 服务器上 conda 环境的 python 绝对路径（非交互式 SSH 不 activate conda，必须用绝对路径）
# 用 `which python` 在 conda 环境激活后查到；这里用 remember 环境的路径
REMOTE_PYTHON = "~/miniconda3/envs/remember/bin/python"
# 需要同步的文件/目录（相对项目根）
SYNC_ITEMS = [
    "wj",                     # WJ 核心包（10 个模块：data/model/training/anticollapse 等）
    "sota",                   # SOTA 对比工具包（sota_comparison.py + sota_results.json 等）
    "experiments",            # 实验脚本（significance_test.py, smoke_test.py）
    "analysis",               # 分析脚本（monitor_collapse, extract_and_plot 等）
    "main.py",                # SBM + 真实数据集主实验
    "run_wj.py",              # WJ 三变体实验运行器（取代旧 compare_with_baselines.py）
    "auto_config.py",         # 图规模自适应 config 选择脚本
    "run_auto_all.py",        # 一键运行 auto_config 全实验
    "deploy.py",              # 部署脚本本身（服务器端也可用）
    "data/cora",
    "data/citeseer",
    "data/Pubmed-Diabetes",
    "data/polblogs_pyg",      # PolBlogs 数据集（PyG 格式）
    "data/polblogs_npz",      # PolBlogs 数据集（npz 格式，无需 PyG）
    "data/amazon_photo",      # Amazon Photo npz（购物网络，17MB）
    "README.md",
    "CHANGELOG.md",
]
# 实验跑完后要拉回的文件（logs/ 和 results/ 子目录）
FETCH_ITEMS = [
    "results/result.md",                # main.py 的产物
    "results/baseline_comparison.md",   # run_wj.py 的产物
    "results/sota_comparison.md",       # sota/sota_comparison.py 的产物
    "results/results_small.json",       # WJ 结果 JSON（断点续传用）
    "results/results_amazon.json",
    "results/results_pubmed.json",
    "image",                            # 图表输出目录
    "logs/main_run.log",                # 主实验日志
    "logs/baseline_small.log",          # baseline 多卡日志
    "logs/baseline_amazon.log",
    "logs/baseline_pubmed.log",
    "logs/merge_baseline.log",          # baseline 合并日志
    "logs/sota_comparison.log",         # SOTA 报告生成日志
    # auto_config.py 的产物
    "results/auto_config_result.md",
    "results/auto_config_decisions.jsonl",
    "results/auto_config_checkpoint.json",
    "logs/auto_config_progress.log",
    "logs/auto_config_run.log",
]
# 项目根目录（本脚本所在目录）
ROOT = Path(__file__).parent.resolve()


# ============================== 工具函数 ==============================
# SSH 通用参数：自动接受新主机密钥，禁用密码交互（配了 key 才能免密）
SSH_OPTS = ["-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=10"]


def ssh(cmd: str, check: bool = True, capture: bool = False) -> str:
    """在服务器上执行命令。返回 stdout（如果 capture=True）。"""
    full = ["ssh"] + SSH_OPTS + [SERVER, f"cd {REMOTE_DIR} && {cmd}"]
    if capture:
        r = subprocess.run(full, capture_output=True, text=True, timeout=60)
        if check and r.returncode != 0:
            print(f"[SSH ERROR] {r.stderr}", file=sys.stderr)
            sys.exit(1)
        return r.stdout.strip()
    else:
        r = subprocess.run(full)
        if check and r.returncode != 0:
            print(f"[SSH ERROR] command failed: {cmd}", file=sys.stderr)
            sys.exit(1)
        return ""


def scp_up(local_path: str, remote_path: str):
    """上传文件/目录到服务器。"""
    subprocess.run(["scp"] + SSH_OPTS + ["-r", local_path, f"{SERVER}:{remote_path}"],
                   check=True)


def scp_down(remote_path: str, local_path: str):
    """从服务器拉回文件/目录。"""
    subprocess.run(["scp"] + SSH_OPTS + ["-r", f"{SERVER}:{remote_path}", local_path],
                   check=True)


# ============================== 子命令 ==============================
def cmd_setup():
    """把代码 + 数据同步到服务器。"""
    print(f"[setup] 同步代码到 {SERVER}:{REMOTE_DIR} ...")
    subprocess.run(["ssh"] + SSH_OPTS + [SERVER, f"mkdir -p {REMOTE_DIR}"],
                   check=True)
    # 在服务器上创建 logs/ 和 results/ 子目录（保持和本地一致的目录结构）
    ssh(f"mkdir -p {REMOTE_DIR}/logs {REMOTE_DIR}/results", check=False, capture=True)
    # 逐个上传：目录先删旧的避免 scp -r 嵌套；文件直接覆盖
    for item in SYNC_ITEMS:
        local = str(ROOT / item)
        if not os.path.exists(local):
            print(f"  [skip] {item} (本地不存在)")
            continue
        remote_item = f"{REMOTE_DIR}/{item}"
        # scp -r source dest_parent/ → 创建 dest_parent/source/
        # 所以 dest_parent 必须是 item 的父目录路径，才能保持目录结构
        # 例：item="data/cora" → remote_parent="~/WJ/data/" → 创建 ~/WJ/data/cora/
        remote_parent = remote_item.rsplit("/", 1)[0] + "/"
        if os.path.isdir(local):
            ssh(f"rm -rf {remote_item}", check=False, capture=True)
            # 确保远程父目录存在
            ssh(f"mkdir -p {remote_parent.rstrip('/')}", check=False, capture=True)
        print(f"  [up] {item} -> {remote_item}")
        scp_up(local, remote_parent)

    print(f"[setup] 完成。远程目录: {REMOTE_DIR}")
    print("[setup] 检查服务器 GPU 和 Python 环境 ...")
    out = ssh(f"{REMOTE_PYTHON} -c 'import torch; print(\"cuda:\", "
              f"torch.cuda.is_available(), \"| device:\", "
              f"torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"cpu\")'",
              check=False, capture=True)
    print(f"  {out}")


def cmd_run(args):
    """远程后台运行脚本（默认 main.py，可通过 --script 指定其他脚本）。

    --script 指定相对于 REMOTE_DIR 的脚本路径（如 analysis/monitor_collapse.py）。
    --log 指定日志文件名（默认 main_run.log；运行诊断脚本时建议改为 diag_run.log）。
    --args 透传给脚本的额外参数（任意脚本都适用，如 auto_config.py 的 "--all --seeds 3"）。
    """
    script = args.script if hasattr(args, 'script') and args.script else "main.py"
    log_file = args.log if hasattr(args, 'log') and args.log else "logs/main_run.log"
    print(f"[run] 远程启动 {script}（后台，日志 -> {log_file}）...")
    # 先杀掉可能在跑的旧进程
    ssh(f"pkill -f 'python {script}' 2>/dev/null; sleep 1; true", check=False)
    # 确保日志目录存在
    ssh(f"mkdir -p logs results", check=False, capture=True)
    # 构造命令行参数：
    # - main.py 专用 flag（--quick/--no-cora 等）只在 main.py 时透传
    # - 任意脚本通用的 --args（如 auto_config.py 的 "--all --seeds 3"）
    main_args = ""
    if script == "main.py":
        if args.quick:
            main_args += " --quick"
        if args.no_cora:
            main_args += " --no-cora"
        if args.no_citeseer:
            main_args += " --no-citeseer"
        if args.no_pubmed:
            main_args += " --no-pubmed"
        if args.no_real:
            main_args += " --no-real"
        if args.no_sig:
            main_args += " --no-sig"
    # 通用 --args 透传：对非 main.py 脚本也生效
    if hasattr(args, 'script_args') and args.script_args:
        main_args += " " + args.script_args
    # 环境变量：限制 OpenBLAS 线程数，避免大图 KMeans 触发 "too many memory regions" 崩溃
    # 服务器 CPU 核数 >128 时 OpenBLAS 默认会尝试分配超过 128 线程导致程序终止
    env_prefix = "OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 "
    # SSH 非交互模式启动后台进程的标准做法：
    # 1) setsid 创建新会话脱离当前 shell
    # 2) 所有 FD（stdin/stdout/stderr）重定向，SSH 才不会挂住等待
    # 3) 外层 bash 也重定向，确保 SSH 连接立即关闭
    cmd = (f"cd {REMOTE_DIR} && setsid bash -c '"
           f"{env_prefix} {REMOTE_PYTHON} -u {script}{main_args} > {log_file} 2>&1 & "
           f"echo $! > main.pid' > /dev/null 2>&1 < /dev/null; "
           f"sleep 1; echo 'started pid:' $(cat {REMOTE_DIR}/main.pid 2>/dev/null)")
    out = ssh(cmd, check=False, capture=True)
    print(f"[run] {out}")
    print(f"[run] 日志: {REMOTE_DIR}/{log_file}")
    print(f"[run] 用 'python deploy.py status' 查看进度")
    print(f"[run] 用 'python deploy.py tail {log_file}' 实时查看日志")


def cmd_run_baseline(args):
    """远程多卡并行运行 run_wj.py。

    服务器有 8 张 RTX A5000（每张 24GB）。将 5 个数据集分成 3 组：
      - GPU 4: cora citeseer polblogs（小图组，5 seeds）
      - GPU 6: amazon_photo（中图，5 seeds）
      - GPU 7: pubmed（大图，2 seeds，铰链关闭）
    三个进程并行跑，互不影响。跑完后合并 JSON 生成最终报告。

    支持断点续传：每个进程用独立的 results JSON，断电后重跑自动跳过已完成的 seed。
    """
    env_prefix = "OMP_NUM_THREADS=64 OPENBLAS_NUM_THREADS=64 MKL_NUM_THREADS=64 "
    # 三个并行进程：每组数据集 + 独立 GPU + 独立 JSON
    jobs = [
        {"gpu": "4", "datasets": "cora citeseer polblogs",
         "json": "results/results_small.json", "log": "logs/baseline_small.log"},
        {"gpu": "6", "datasets": "amazon_photo",
         "json": "results/results_amazon.json", "log": "logs/baseline_amazon.log"},
        {"gpu": "7", "datasets": "pubmed",
         "json": "results/results_pubmed.json", "log": "logs/baseline_pubmed.log"},
    ]
    print("[run_baseline] 多卡并行启动 run_wj.py")
    print(f"  服务器: {SERVER}, 3 个进程分别用 GPU 4/6/7")
    print(f"  断点续传: 每个进程独立 JSON，断电后重跑自动跳过已完成的 seed")
    # 确保日志和结果目录存在
    ssh("mkdir -p logs results", check=False, capture=True)
    # 先杀掉可能在跑的旧进程
    ssh("pkill -f 'run_wj.py' 2>/dev/null; sleep 1; true", check=False)
    for j in jobs:
        script_args = (f"--datasets {j['datasets']} "
                        f"--results-json {j['json']} "
                        f"--output results/baseline_{j['gpu']}.md")
        cmd = (f"cd {REMOTE_DIR} && setsid bash -c '"
               f"{env_prefix} CUDA_VISIBLE_DEVICES={j['gpu']} "
               f"{REMOTE_PYTHON} -u run_wj.py {script_args} "
               f"> {j['log']} 2>&1 & echo $! > pid_{j['gpu']}.pid' "
               f"> /dev/null 2>&1 < /dev/null; sleep 0.5")
        ssh(cmd, check=False)
        print(f"  [GPU {j['gpu']}] {j['datasets']} -> {j['log']} (json: {j['json']})")
    print(f"\n[run_baseline] 3 个进程已启动（后台并行）")
    print(f"  用 'python deploy.py status_baseline' 查看进度")
    print(f"  用 'python deploy.py tail_baseline' 实时查看日志")
    print(f"  完成后用 'python deploy.py fetch_baseline' 拉回结果")


def cmd_status_baseline():
    """查看多卡并行 baseline 实验进度。"""
    jobs = [
        {"gpu": "4", "log": "logs/baseline_small.log", "pid": "pid_4.pid"},
        {"gpu": "6", "log": "logs/baseline_amazon.log", "pid": "pid_6.pid"},
        {"gpu": "7", "log": "logs/baseline_pubmed.log", "pid": "pid_7.pid"},
    ]
    all_done = True
    for j in jobs:
        out = ssh(f"ps -p $(cat {j['pid']} 2>/dev/null) -o etime,cmd 2>/dev/null "
                  f"|| echo 'DONE'", check=False, capture=True)
        if "DONE" in out:
            print(f"[GPU {j['gpu']}] 完成")
            # 显示最后 3 行日志
            tail = ssh(f"tail -3 {j['log']} 2>/dev/null", check=False, capture=True)
            if tail:
                for line in tail.split("\n"):
                    print(f"  {line}")
        else:
            all_done = False
            print(f"[GPU {j['gpu']}] 运行中:")
            # 取第一行简短信息
            first_line = out.split("\n")[1] if len(out.split("\n")) > 1 else out
            print(f"  {first_line}")
            tail = ssh(f"tail -1 {j['log']} 2>/dev/null", check=False, capture=True)
            if tail:
                print(f"  最新: {tail}")
    if all_done:
        print("\n[status] 全部完成！可用 'python deploy.py fetch_baseline' 拉回结果")
    else:
        print("\n[status] 仍有进程在跑...")


def cmd_tail_baseline():
    """实时查看 baseline 日志（合并 3 个进程的日志尾部）。"""
    logs = ["logs/baseline_small.log", "logs/baseline_amazon.log", "logs/baseline_pubmed.log"]
    print("[tail_baseline] 实时显示 3 个进程日志尾部（Ctrl+C 退出）:")
    while True:
        print(f"\n{'='*60} {time.strftime('%H:%M:%S')} {'='*60}")
        for log in logs:
            tail = ssh(f"tail -2 {log} 2>/dev/null", check=False, capture=True)
            if tail:
                print(f"\n--- {log} ---")
                print(tail)
        time.sleep(10)


def cmd_fetch_baseline():
    """拉回 baseline 实验结果（3 个 JSON + 3 个 md + 3 个 log + image/ + SOTA 报告）。"""
    items = [
        "results/results_small.json", "results/results_amazon.json", "results/results_pubmed.json",
        "results/baseline_4.md", "results/baseline_6.md", "results/baseline_7.md",
        "logs/baseline_small.log", "logs/baseline_amazon.log", "logs/baseline_pubmed.log",
        "results/baseline_comparison.md", "image",
        "results/sota_comparison.md",   # SOTA 对比报告（cmd_merge_baseline 末尾生成）
        "logs/sota_comparison.log",     # SOTA 报告生成日志
        "logs/merge_baseline.log",      # baseline 合并日志
    ]
    print("[fetch_baseline] 拉回 baseline 实验结果 ...")
    for item in items:
        remote = f"{REMOTE_DIR}/{item}"
        local = str(ROOT / item)
        local_dir = os.path.dirname(local) if "." in os.path.basename(local) else local
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        r = subprocess.run(["scp", "-r", f"{SERVER}:{remote}", local],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  [down] {item}")
        else:
            print(f"  [skip] {item} ({r.stderr.strip()[:80]})")
    print("[fetch_baseline] 完成")


def cmd_merge_baseline():
    """远程合并 3 个 JSON 并生成最终 baseline_comparison.md + 图表 + SOTA 对比报告。"""
    print("[merge_baseline] 远程合并 3 个 JSON ...")
    cmd = (f"cd {REMOTE_DIR} && {REMOTE_PYTHON} -u run_wj.py "
           f"--merge results/results_small.json results/results_amazon.json results/results_pubmed.json "
           f"--output results/baseline_comparison.md > logs/merge_baseline.log 2>&1; "
           f"echo 'merge exit:' $?")
    out = ssh(cmd, check=False, capture=True)
    print(f"  {out}")
    # 生成 SOTA 对比报告（merge baseline 后立刻跑，结果一并拉回）
    # 依赖：sota/sota_results.json + 3 个 results JSON（含 acc 字段，刚跑完）
    # 传 3 个 JSON 让 sota/sota_comparison.py 合并，PubMed 结果在 results_pubmed.json 中
    print("[merge_baseline] 生成 SOTA 对比报告 ...")
    sota_cmd = (f"cd {REMOTE_DIR} && {REMOTE_PYTHON} -u sota/sota_comparison.py "
                f"--our-json results/results_small.json results/results_amazon.json results/results_pubmed.json "
                f"--sota-json sota/sota_results.json "
                f"--output results/sota_comparison.md > logs/sota_comparison.log 2>&1; "
                f"echo 'sota exit:' $?")
    sota_out = ssh(sota_cmd, check=False, capture=True)
    print(f"  {sota_out}")
    # 拉回合并后的结果（含 SOTA 报告）
    cmd_fetch_baseline()
    # 显示合并日志
    log = ssh("cat logs/merge_baseline.log 2>/dev/null", check=False, capture=True)
    if log:
        print("\n[merge_baseline] 合并日志:")
        for line in log.split("\n"):
            print(f"  {line}")
    # 显示 SOTA 报告生成日志
    sota_log = ssh("cat logs/sota_comparison.log 2>/dev/null", check=False, capture=True)
    if sota_log:
        print("\n[merge_baseline] SOTA 报告日志:")
        for line in sota_log.split("\n"):
            print(f"  {line}")


def cmd_status(args=None):
    """查看远程实验是否在跑 + 当前进度。"""
    log_file = "logs/main_run.log"
    if args is not None and hasattr(args, 'log') and args.log:
        log_file = args.log
    # 检查进程
    out = ssh("ps -p $(cat main.pid 2>/dev/null) -o pid,etime,cmd 2>/dev/null "
              "|| echo 'NOT_RUNNING'", check=False, capture=True)
    if "NOT_RUNNING" in out:
        print("[status] 实验未在运行（可能已完成或未启动）")
        # 检查 result.md 是否存在
        r = ssh("test -f results/result.md && echo 'result.md EXISTS' || echo 'no result.md'",
                check=False, capture=True)
        print(f"[status] {r}")
    else:
        print(f"[status] 实验运行中:")
        print(f"  {out}")
        # 显示日志最后 5 行
        tail = ssh(f"tail -5 {log_file} 2>/dev/null", check=False, capture=True)
        if tail:
            print(f"[status] 最新日志:")
            for line in tail.split("\n"):
                print(f"  {line}")


def cmd_tail(args=None):
    """实时查看远程日志（Ctrl+C 退出）。"""
    log_file = "logs/main_run.log"
    if args is not None and hasattr(args, 'log') and args.log:
        log_file = args.log
    # 兼容旧调用：cmd_tail() 无参数时用默认日志
    if args is not None and hasattr(args, 'log_file'):
        log_file = args.log_file
    print(f"[tail] 实时日志 {log_file}（Ctrl+C 退出）:")
    subprocess.run(["ssh"] + SSH_OPTS + [SERVER, f"cd {REMOTE_DIR} && tail -f {log_file}"])


def cmd_fetch():
    """拉回 results/ + image/ + logs/ 中的结果文件。"""
    print(f"[fetch] 拉回结果到本地 ...")
    for item in FETCH_ITEMS:
        remote = f"{REMOTE_DIR}/{item}"
        local = str(ROOT / item)
        # 确保本地目标目录存在（image/ 等）
        local_dir = os.path.dirname(local) if "." in os.path.basename(local) else local
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)
        print(f"  [down] {item} -> {local}")
        # scp 失败不致命（比如 result.md 还没生成）
        r = subprocess.run(["scp", "-r", f"{SERVER}:{remote}", local],
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    [warn] {r.stderr.strip()}")
    # 额外拉回 logs/ 下所有 *.log 日志文件（diag_run.log 等未在 FETCH_ITEMS 中的）
    logs_out = ssh("ls logs/*.log 2>/dev/null", check=False, capture=True)
    if logs_out:
        for log_file in logs_out.split("\n"):
            log_file = log_file.strip()
            if not log_file or log_file == "logs/main_run.log":
                continue  # main_run.log 已在 FETCH_ITEMS
            remote = f"{REMOTE_DIR}/{log_file}"
            local = str(ROOT / log_file)
            print(f"  [down] {log_file} -> {local}")
            r = subprocess.run(["scp", "-r", f"{SERVER}:{remote}", local],
                               capture_output=True, text=True)
            if r.returncode != 0:
                print(f"    [warn] {r.stderr.strip()}")
    print(f"[fetch] 完成。本地查看: results/, image/, logs/")


def cmd_all(args):
    """一键完成：setup + run + 等待 + fetch。"""
    log_file = args.log if hasattr(args, 'log') and args.log else "logs/main_run.log"
    cmd_setup()
    cmd_run(args)
    print("\n[all] 等待实验完成（每 60 秒检查一次，Ctrl+C 中断等待但不影响远程实验）...")

    # 启动后立即做一次健康检查：进程秒退通常是脚本路径错误或 import 失败。
    # 不做这步的话，run 后 sleep(60) 才发现进程已死，会误判成"实验完成"。
    time.sleep(3)  # 给进程 3 秒完成 import 和初始化
    early_check = ssh("ps -p $(cat main.pid 2>/dev/null) > /dev/null 2>&1 "
                      "&& echo 'RUNNING' || echo 'DEAD'", check=False, capture=True)
    if "DEAD" in early_check:
        # 进程 3 秒内就死了，几乎肯定是启动失败。拉回日志头部帮助定位。
        print("\n[all] [错误] 进程启动后立即退出！可能是脚本路径错误或 import 失败。")
        log_head = ssh(f"head -30 {log_file} 2>/dev/null", check=False, capture=True)
        if log_head:
            print(f"[all] 日志前 30 行:")
            for line in log_head.split("\n"):
                print(f"  {line}")
        print("[all] 已中止，未执行 fetch。请检查 SYNC_ITEMS 是否包含该脚本。")
        return  # 直接返回，不 fetch 误导用户

    while True:
        time.sleep(60)
        out = ssh("ps -p $(cat main.pid 2>/dev/null) > /dev/null 2>&1 "
                  "&& echo 'RUNNING' || echo 'DONE'", check=False, capture=True)
        if "DONE" in out:
            print("\n[all] 实验完成！")
            break
        print(f"[all] 仍在运行... {time.strftime('%H:%M:%S')}")
        tail = ssh(f"tail -1 {log_file} 2>/dev/null", check=False, capture=True)
        if tail:
            print(f"      {tail}")
    cmd_fetch()
    print("\n[all] 全部完成！查看 results/result.md 了解结果。")


# ============================== 入口 ==============================
def main():
    parser = argparse.ArgumentParser(description="部署实验到服务器并运行")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("setup", help="传代码 + 数据到服务器")

    p_status = sub.add_parser("status", help="查看远程实验进度")
    p_status.add_argument("--log", default=None,
                          help="日志文件名（默认 logs/main_run.log）")

    p_tail = sub.add_parser("tail", help="实时查看远程日志")
    p_tail.add_argument("log_file", nargs="?", default="logs/main_run.log",
                        help="日志文件名（默认 logs/main_run.log）")
    p_tail.add_argument("--log", default=None,
                        help="同上，用于 --log logs/diag_run.log 形式")

    sub.add_parser("fetch", help="拉回 results/ + image/ + logs/")

    p_run = sub.add_parser("run", help="远程后台运行脚本（默认 main.py）")
    p_run.add_argument("--script", default=None,
                       help="脚本路径（相对 REMOTE_DIR，如 analysis/monitor_collapse.py）")
    p_run.add_argument("--log", default=None,
                       help="日志文件名（默认 logs/main_run.log；诊断脚本建议 logs/diag_run.log）")
    p_run.add_argument("--args", dest="script_args", default="",
                       help="透传给脚本的参数（如 auto_config.py 的 \"--all --seeds 3\"）")
    p_run.add_argument("--quick", action="store_true", help="快速版（1 seed，仅 main.py）")
    p_run.add_argument("--no-cora", action="store_true", help="跳过 Cora（仅 main.py）")
    p_run.add_argument("--no-citeseer", action="store_true", help="跳过 CiteSeer（仅 main.py）")
    p_run.add_argument("--no-pubmed", action="store_true",
                       help="跳过 Pubmed（大图较慢，仅 main.py）")
    p_run.add_argument("--no-real", action="store_true", help="跳过所有真实数据集（仅 main.py）")
    p_run.add_argument("--no-sig", action="store_true", help="跳过显著性检验（仅 main.py）")

    p_all = sub.add_parser("all", help="一键：setup + run + 等待 + fetch")
    p_all.add_argument("--script", default=None,
                       help="脚本路径（相对 REMOTE_DIR，不指定则跑 main.py）")
    p_all.add_argument("--log", default=None,
                       help="日志文件名（默认 logs/main_run.log）")
    p_all.add_argument("--args", dest="script_args", default="",
                       help="透传给脚本的参数（如 auto_config.py 的 \"--all --seeds 3\"）")
    p_all.add_argument("--quick", action="store_true", help="快速版（1 seed）")
    p_all.add_argument("--no-cora", action="store_true", help="跳过 Cora")
    p_all.add_argument("--no-citeseer", action="store_true", help="跳过 CiteSeer")
    p_all.add_argument("--no-pubmed", action="store_true",
                       help="跳过 Pubmed（大图较慢）")
    p_all.add_argument("--no-real", action="store_true", help="跳过所有真实数据集")
    p_all.add_argument("--no-sig", action="store_true", help="跳过显著性检验")

    # ---- baseline 多卡并行子命令 ----
    sub.add_parser("run_baseline", help="多卡并行跑 run_wj.py（GPU 4/6/7）")
    sub.add_parser("status_baseline", help="查看多卡 baseline 实验进度")
    sub.add_parser("tail_baseline", help="实时查看 baseline 日志（Ctrl+C 退出）")
    sub.add_parser("fetch_baseline", help="拉回 baseline 结果（JSON + md + log + image）")
    sub.add_parser("merge_baseline", help="远程合并 3 个 JSON 并生成最终报告")

    # ---- baseline 一键到底：setup + run_baseline + 等待 + merge + fetch ----
    p_baseline_all = sub.add_parser("all_baseline",
                                    help="一键：setup + run_baseline + 等待 + merge + fetch")
    p_baseline_all.add_argument("--no-setup", action="store_true",
                                help="跳过 setup（代码已同步时用）")

    args = parser.parse_args()
    if args.cmd == "setup":
        cmd_setup()
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "status":
        cmd_status(args)
    elif args.cmd == "tail":
        # 兼容位置参数和 --log 两种形式
        if hasattr(args, 'log') and args.log:
            args.log_file = args.log
        cmd_tail(args)
    elif args.cmd == "fetch":
        cmd_fetch()
    elif args.cmd == "all":
        cmd_all(args)
    elif args.cmd == "run_baseline":
        cmd_run_baseline(args)
    elif args.cmd == "status_baseline":
        cmd_status_baseline()
    elif args.cmd == "tail_baseline":
        cmd_tail_baseline()
    elif args.cmd == "fetch_baseline":
        cmd_fetch_baseline()
    elif args.cmd == "merge_baseline":
        cmd_merge_baseline()
    elif args.cmd == "all_baseline":
        if not args.no_setup:
            cmd_setup()
        cmd_run_baseline(args)
        print("\n[all_baseline] 等待 3 个进程完成（每 30 秒检查一次）...")
        jobs = [{"gpu": "4", "pid": "pid_4.pid"},
                {"gpu": "6", "pid": "pid_6.pid"},
                {"gpu": "7", "pid": "pid_7.pid"}]
        # 健康检查
        time.sleep(5)
        for j in jobs:
            chk = ssh(f"ps -p $(cat {j['pid']} 2>/dev/null) > /dev/null 2>&1 "
                      f"&& echo 'RUNNING' || echo 'DEAD'", check=False, capture=True)
            if "DEAD" in chk:
                print(f"  [GPU {j['gpu']}] 启动后立即退出！查看日志:")
                log = ssh(f"head -10 logs/baseline_{'small' if j['gpu']=='4' else 'amazon' if j['gpu']=='6' else 'pubmed'}.log 2>/dev/null",
                          check=False, capture=True)
                if log:
                    for line in log.split("\n"):
                        print(f"    {line}")
        # 轮询等待
        while True:
            time.sleep(30)
            all_done = True
            for j in jobs:
                out = ssh(f"ps -p $(cat {j['pid']} 2>/dev/null) > /dev/null 2>&1 "
                          f"&& echo 'RUNNING' || echo 'DONE'", check=False, capture=True)
                if "RUNNING" in out:
                    all_done = False
                    tail = ssh(f"tail -1 logs/baseline_{'small' if j['gpu']=='4' else 'amazon' if j['gpu']=='6' else 'pubmed'}.log 2>/dev/null",
                               check=False, capture=True)
                    print(f"  [GPU {j['gpu']}] 运行中: {tail}")
            if all_done:
                print("\n[all_baseline] 全部完成！开始合并...")
                break
        cmd_merge_baseline()
        print("\n[all_baseline] 全部完成！查看 results/baseline_comparison.md 了解结果。")


if __name__ == "__main__":
    main()
