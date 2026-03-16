#!/usr/bin/env bash
set -euo pipefail

# 顺序训练队列：BEAST -> SIREN -> OFT -> FAST
# 用法：
#   ./scripts/run_aloha_vla_queue.sh
#   CONTINUE_ON_FAIL=1 ./scripts/run_aloha_vla_queue.sh   # 失败也继续后续
#   ENV_NAME=beast_cal ./scripts/run_aloha_vla_queue.sh    # 指定conda环境名
#   CUDA_VISIBLE_DEVICES=0,1,2,3 ./scripts/run_aloha_vla_queue.sh
#   ./scripts/run_aloha_vla_queue.sh root_data_dir=/data1/... devices=4  # 额外Hydra override(会作用到每个训练)

ENV_NAME=${ENV_NAME:-beast_cal}
CONTINUE_ON_FAIL=${CONTINUE_ON_FAIL:-0}

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TS=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${REPO_ROOT}/logs/train_queue/${TS}"
mkdir -p "${LOG_DIR}"

# 尽量稳妥地找到 conda
conda_cmd="conda"
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "${HOME}/miniconda3/condabin/conda" ]]; then
    conda_cmd="${HOME}/miniconda3/condabin/conda"
  elif [[ -x "${HOME}/miniconda3/bin/conda" ]]; then
    conda_cmd="${HOME}/miniconda3/bin/conda"
  else
    echo "[FATAL] 找不到 conda。请确保 conda 在 PATH，或安装在 ~/miniconda3" >&2
    exit 127
  fi
fi

run_one() {
  local name="$1"; shift
  local script_rel="$1"; shift

  local script_path="${REPO_ROOT}/${script_rel}"
  local log_file="${LOG_DIR}/${name}.log"

  if [[ ! -f "${script_path}" ]]; then
    echo "[FATAL] 脚本不存在: ${script_path}" | tee -a "${log_file}" >&2
    return 127
  fi

  echo "==================================================================" | tee -a "${log_file}"
  echo "[QUEUE] START ${name} @ $(date -Iseconds)" | tee -a "${log_file}"
  echo "[QUEUE] CMD: ${conda_cmd} run -n ${ENV_NAME} --no-capture-output python ${script_path} $*" | tee -a "${log_file}"

  local start_epoch
  start_epoch=$(date +%s)

  # NOTE: --no-capture-output 让 stdout/stderr 实时输出到 tmux，并可被 tee 记录
  set +e
  "${conda_cmd}" run -n "${ENV_NAME}" --no-capture-output \
    python "${script_path}" "$@" 2>&1 | tee -a "${log_file}"
  local exit_code=${PIPESTATUS[0]}
  set -e

  local end_epoch
  end_epoch=$(date +%s)
  local dur=$((end_epoch - start_epoch))

  echo "[QUEUE] END   ${name} @ $(date -Iseconds) exit=${exit_code} dur=${dur}s" | tee -a "${log_file}"

  return ${exit_code}
}

common_overrides=("$@")

# 四个 ALOHA 训练入口
jobs=(
  # "FAST|beast/training_aloha_fast.py"
  "SIREN|beast/training_aloha_siren.py"
  "BEAST|beast/training_aloha.py"
  "OFT|beast/training_aloha_oft.py"
)

fail_count=0
for item in "${jobs[@]}"; do
  IFS='|' read -r name script_rel <<<"${item}"

  if run_one "${name}" "${script_rel}" "${common_overrides[@]}"; then
    echo "[QUEUE] OK ${name}"
  else
    ec=$?
    echo "[QUEUE] FAIL ${name} exit=${ec} (log: ${LOG_DIR}/${name}.log)" >&2
    fail_count=$((fail_count + 1))

    if [[ "${CONTINUE_ON_FAIL}" != "1" ]]; then
      echo "[QUEUE] 停止队列（如需失败也继续：CONTINUE_ON_FAIL=1）" >&2
      exit ${ec}
    fi
  fi

done

echo "[QUEUE] ALL DONE @ $(date -Iseconds) fail_count=${fail_count}"