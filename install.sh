#!/bin/bash
# RVC翻唱工具 - 一键安装脚本（Ubuntu虚拟环境方案）
# 适用于Android Termux环境
# 原理: 在Termux中用proot-distro安装一个虚拟Ubuntu系统，
#       RVC跑在Ubuntu里，所有依赖都能正常pip安装

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo "=========================================="
echo "  RVC翻唱工具 - 安装脚本"
echo "=========================================="
echo ""

# 检查是否在Termux中运行
if [ -z "$TERMUX_VERSION" ]; then
    echo -e "${YELLOW}警告: 此脚本专为Termux设计，可能在其他环境中无法正常工作${NC}"
fi

# 代码目录 = 本脚本所在目录（git clone 下来直接 bash install.sh 即可，不再要求 ~/rvc-app）
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$CODE_DIR/server.py" ]; then
    echo -e "${RED}错误: 在 $CODE_DIR 下找不到 server.py${NC}"
    echo "请在仓库根目录运行: bash install.sh"
    exit 1
fi
echo "代码目录: $CODE_DIR"

# 1. 更新包管理器并安装proot-distro
echo -e "${GREEN}[1/6] 更新包管理器并安装proot-distro...${NC}"
pkg update -y
pkg install -y proot-distro curl tar coreutils

# 2. 安装Ubuntu虚拟系统（若已安装则跳过）
echo -e "${GREEN}[2/6] 检查Ubuntu虚拟系统...${NC}"
# proot-distro 目录兼容检查（新版5.x用 containers/<名>/rootfs，旧版用 installed-rootfs/<名>）
UBUNTU_ROOTFS=""
for d in \
    "$PREFIX/var/lib/proot-distro/containers/ubuntu/rootfs" \
    "$PREFIX/var/lib/proot-distro/installed-rootfs/ubuntu"; do
    if [ -d "$d" ]; then
        UBUNTU_ROOTFS="$d"
        break
    fi
done

if [ -z "$UBUNTU_ROOTFS" ]; then
    # 优先使用手机Download里已下载好的本地镜像（国内连Docker Hub会超时）
    LOCAL_ROOTFS=""
    for f in "$HOME"/storage/shared/Download/ubuntu-*.tar.xz "$HOME"/storage/shared/Download/ubuntu-*.tar.gz; do
        [ -f "$f" ] && LOCAL_ROOTFS="$f"
    done
    if [ -n "$LOCAL_ROOTFS" ]; then
        echo "检测到本地Ubuntu镜像: $(basename "$LOCAL_ROOTFS")"
        echo "正在导入（需要几分钟，请耐心等待）..."
        # 新版proot-distro(5.x)语法: install -n <容器名> <本地rootfs文件>
        # 必须指定 -n ubuntu，否则容器名会变成文件名，后续login会找不到
        proot-distro install -n ubuntu "$LOCAL_ROOTFS"
    else
        echo "未找到本地镜像，尝试在线下载Ubuntu系统（约500MB）"
        echo "提示: 国内网络下载Docker Hub可能超时。如果超时，请先在电脑上下载:"
        echo "  https://cloud-images.ubuntu.com/releases/jammy/release-20260320/ubuntu-22.04-server-cloudimg-arm64-root.tar.xz"
        echo "把文件放到手机Download目录后，重新运行本脚本即可自动导入"
        proot-distro install ubuntu
    fi
    # 重新确定容器路径（新安装的容器目录）
    UBUNTU_ROOTFS=""
    for d in \
        "$PREFIX/var/lib/proot-distro/containers/ubuntu/rootfs" \
        "$PREFIX/var/lib/proot-distro/installed-rootfs/ubuntu"; do
        if [ -d "$d" ]; then
            UBUNTU_ROOTFS="$d"
            break
        fi
    done
    if [ -z "$UBUNTU_ROOTFS" ]; then
        echo -e "${RED}错误: Ubuntu安装后找不到容器目录，安装中止${NC}"
        exit 1
    fi
else
    echo "Ubuntu已安装，跳过"
fi

# 3. 将代码复制到Ubuntu容器内
# 重要: proot-distro登录后，容器内的 /root 就是容器rootfs的 /root，
#       不会自动共享Termux的home目录，所以代码必须复制进容器
echo -e "${GREEN}[3/6] 将代码复制到Ubuntu中...${NC}"
# 确保容器内有 /root 目录
mkdir -p "$UBUNTU_ROOTFS/root"
# 重装前先把容器里已有的权重挪出来，避免每次重跑都重新下载 180MB
KEEP_DIR="$UBUNTU_ROOTFS/root/.rvc-weights-keep"
rm -rf "$KEEP_DIR"
for w in assets/hubert/hubert_base.pt assets/rmvpe/rmvpe.pt; do
    if [ -f "$UBUNTU_ROOTFS/root/rvc-app/$w" ]; then
        mkdir -p "$KEEP_DIR/${w%/*}"
        mv "$UBUNTU_ROOTFS/root/rvc-app/$w" "$KEEP_DIR/$w"
    fi
done
rm -rf "$UBUNTU_ROOTFS/root/rvc-app"
mkdir -p "$UBUNTU_ROOTFS/root/rvc-app"
# 用 tar 管道复制：排除 .git（含完整历史对象，几十 MB）、本机缓存和运行时文件
tar -C "$CODE_DIR" \
    --exclude=.git --exclude=__pycache__ --exclude='*.pyc' \
    --exclude=uploads --exclude=outputs --exclude=server.log \
    -cf - . | tar -C "$UBUNTU_ROOTFS/root/rvc-app" -xf -
# 权重放回原位（源码里没有的话）
for w in assets/hubert/hubert_base.pt assets/rmvpe/rmvpe.pt; do
    if [ ! -f "$UBUNTU_ROOTFS/root/rvc-app/$w" ] && [ -f "$KEEP_DIR/$w" ]; then
        mkdir -p "$UBUNTU_ROOTFS/root/rvc-app/${w%/*}"
        mv "$KEEP_DIR/$w" "$UBUNTU_ROOTFS/root/rvc-app/$w"
    fi
done
rm -rf "$KEEP_DIR"
# 清掉运行时目录（上传/输出文件由服务器运行时自动创建）
rm -rf "$UBUNTU_ROOTFS/root/rvc-app/uploads" "$UBUNTU_ROOTFS/root/rvc-app/outputs"

# 4. 预训练权重（hubert_base.pt 必需，rmvpe.pt 可选）
#    权重不进 git（单个 180MB 左右，超 GitHub 限制），所以这里补齐
echo -e "${GREEN}[4/6] 检查预训练权重...${NC}"
HUBERT_DIR="$UBUNTU_ROOTFS/root/rvc-app/assets/hubert"
RMVPE_DIR="$UBUNTU_ROOTFS/root/rvc-app/assets/rmvpe"
mkdir -p "$HUBERT_DIR" "$RMVPE_DIR"

# 完整性判断：两个权重都约 180MB，>150MB 视为下完（防止拿到 HTML 错误页）
is_complete_pt() {
    [ -f "$1" ] || return 1
    local sz
    sz=$(stat -c%s "$1" 2>/dev/null || echo 0)
    [ "$sz" -gt 150000000 ]
}

# 在 Termux 侧下载：容器内 DNS/网络经常不通，下好再放进 rootfs 更稳
fetch_pt() {
    local dest="$1"; shift
    for url in "$@"; do
        echo "  尝试: $url"
        if curl -fL --connect-timeout 20 --retry 2 --retry-delay 2 -o "$dest.part" "$url"; then
            if is_complete_pt "$dest.part"; then
                mv -f "$dest.part" "$dest"
                return 0
            fi
            echo -e "  ${YELLOW}文件不完整，换下一个源${NC}"
        fi
        rm -f "$dest.part"
    done
    return 1
}

HUBERT_DST="$HUBERT_DIR/hubert_base.pt"
if is_complete_pt "$HUBERT_DST"; then
    echo "hubert_base.pt 已存在，跳过"
elif is_complete_pt "$CODE_DIR/assets/hubert/hubert_base.pt"; then
    echo "从代码目录复制 hubert_base.pt..."
    cp -f "$CODE_DIR/assets/hubert/hubert_base.pt" "$HUBERT_DST"
elif is_complete_pt "$HOME/storage/shared/Download/hubert_base.pt"; then
    echo "从手机 Download 目录复制 hubert_base.pt..."
    cp -f "$HOME/storage/shared/Download/hubert_base.pt" "$HUBERT_DST"
else
    echo "下载 hubert_base.pt（约 181 MB，需要几分钟）..."
    if ! fetch_pt "$HUBERT_DST" \
        "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt" \
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"; then
        echo -e "${RED}错误: hubert_base.pt 下载失败（断网或源不可达）${NC}"
        echo "  它是必需文件，缺少时无法转换。可以手动下载后二选一："
        echo "    1) 放到 $HUBERT_DST"
        echo "    2) 放到手机 Download 目录（文件名 hubert_base.pt）后重跑本脚本"
        echo "  下载地址: https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt"
        exit 1
    fi
fi

RMVPE_DST="$RMVPE_DIR/rmvpe.pt"
if is_complete_pt "$RMVPE_DST"; then
    echo "rmvpe.pt 已存在，跳过"
elif [ "${INSTALL_RMVPE:-0}" = "1" ]; then
    echo "下载 rmvpe.pt（可选，约 173 MB）..."
    fetch_pt "$RMVPE_DST" \
        "https://hf-mirror.com/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt" \
        "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt" \
        || echo -e "${YELLOW}警告: rmvpe.pt 下载失败，不影响使用（默认用 harvest 提取音高）${NC}"
else
    echo "rmvpe.pt 未安装（可选：INSTALL_RMVPE=1 bash install.sh 一并下载）"
fi

# 5. 生成Ubuntu内的环境安装脚本（写入容器rootfs，登录后容器内可见）
echo -e "${GREEN}[5/6] 在Ubuntu中安装环境（软件源/依赖，需要几分钟）...${NC}"
cat > "$UBUNTU_ROOTFS/root/rvc_setup_ubuntu.sh" << 'SETUP'
#!/bin/bash
set -e
export DEBIAN_FRONTEND=noninteractive

# 确保DNS可用（proot环境可能没有resolv.conf）
if [ ! -s /etc/resolv.conf ]; then
    printf "nameserver 223.5.5.5\nnameserver 114.114.114.114\n" > /etc/resolv.conf
fi

# 清除Android环境变量（proot会从Termux继承），
# 否则pip的platformdirs会误判为Android系统并报错
unset ANDROID_ROOT ANDROID_DATA ANDROID_STORAGE \
      ANDROID_ART_ROOT ANDROID_DEXPREOPT_ROOT ANDROID_TZDATA_ROOT 2>/dev/null || true

echo "[Ubuntu] 配置国内软件源（加速下载）..."
if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then
    sed -i \
        -e 's@http://ports.ubuntu.com/ubuntu-ports@http://mirrors.huaweicloud.com/ubuntu-ports@g' \
        -e 's@http://[a-z.]*archive\.ubuntu\.com/ubuntu@http://mirrors.huaweicloud.com/ubuntu@g' \
        -e 's@http://security\.ubuntu\.com/ubuntu@http://mirrors.huaweicloud.com/ubuntu@g' \
        /etc/apt/sources.list.d/ubuntu.sources
else
    sed -i \
        -e 's@http://ports.ubuntu.com/ubuntu-ports@http://mirrors.huaweicloud.com/ubuntu-ports@g' \
        -e 's@http://[a-z.]*archive\.ubuntu\.com/ubuntu@http://mirrors.huaweicloud.com/ubuntu@g' \
        -e 's@http://security\.ubuntu\.com/ubuntu@http://mirrors.huaweicloud.com/ubuntu@g' \
        /etc/apt/sources.list
fi

echo "[Ubuntu] 更新软件源..."
apt update -y

echo "[Ubuntu] 安装Python 3.10和基础工具..."
apt install -y python3.10 python3-pip ffmpeg libsndfile1 build-essential

echo "[Ubuntu] 安装Python依赖（torch较大，需要几分钟）..."
# 固定pip为24.0: 新版pip(24.1+)会拒绝omegaconf旧版metadata
# （`PyYAML (>=5.1.*)` 写法），导致fairseq依赖无法解析
pip3 install "pip==24.0" -i https://mirrors.huaweicloud.com/repository/pypi/simple
# --ignore-installed: Ubuntu系统自带的blinker等旧包是distutils安装的，
# pip无法卸载它们（Cannot uninstall 'blinker'），直接覆盖安装新版本
# 使用华为云pip镜像加速国内下载
pip3 install --ignore-installed -i https://mirrors.huaweicloud.com/repository/pypi/simple \
    -r /root/rvc-app/requirements.txt

echo "[Ubuntu] 环境准备完成"
SETUP

# 5. 在Ubuntu中执行安装
proot-distro login ubuntu -- bash /root/rvc_setup_ubuntu.sh
rm -f "$UBUNTU_ROOTFS/root/rvc_setup_ubuntu.sh"

# 6. 创建必要目录（容器内）
echo -e "${GREEN}[6/6] 创建必要目录...${NC}"
mkdir -p "$UBUNTU_ROOTFS/root/rvc-app/models" \
         "$UBUNTU_ROOTFS/root/rvc-app/uploads" \
         "$UBUNTU_ROOTFS/root/rvc-app/outputs"

# 完成
echo -e "${GREEN}安装完成！${NC}"
echo "=========================================="
echo ""
echo "接下来（打包分发给客户）:"
echo "  1. 运行备份脚本打包整个环境:"
echo "     bash $CODE_DIR/backup.sh"
echo "  2. 打包好的文件在 手机/Download 目录，连同 Termux 安装包和"
echo "     restore.sh 一起发给客户即可"
echo ""
echo "自己测试（可选）:"
echo "  1. 将你的RVC模型文件(.pth)放到 $CODE_DIR/models/ 目录后"
echo "     重新运行本脚本（代码会重新同步进Ubuntu）"
echo "  2. 运行: bash $CODE_DIR/start.sh"
echo "  3. 浏览器打开: http://localhost:8080"
echo ""
echo "如果遇到问题:"
echo "  - 需要至少4GB可用存储空间（Ubuntu约1.5GB + 权重约360MB + 模型）"
echo "  - 需要稳定的网络（约1.7GB下载，走 hf-mirror 镜像）"
echo "  - 安装中断可重新运行本脚本（已完成的步骤会跳过）"
echo ""
