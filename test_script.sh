#!/bin/bash

# 自动循环运行回测，每轮57分钟，暂停3分钟

# 无限循环
while true
do
    echo "🚀 开始运行 main.py ..."
    python main_evaluator.py &
    PID=$!

    # 运行57分钟（57*60秒）
    sleep $((57 * 60))

    echo "⏹ 停止 main.py (PID=$PID)..."
    kill $PID

    # 等待3分钟
    echo "🕒 等待3分钟后重新开始..."
    sleep $((3 * 60))
done

