# 回放题第一轮：状态（2026-11-16）

10 道回放题（D-041）的原始结果，**尚未判分**。

| 目录 | 内容 | 有效 |
|---|---|---|
| `2026-11-16-baseline-replay-thinking/` | 弱基线，开思考，每题 3 次 | 30 次里 27 次有结果；2 次思考用完 65536 输出没给答案，1 次（basis 第 2 次）碰上余额不足，缺 |
| `2026-11-16-baseline-replay-nothink/` | 弱基线，关思考，每题 3 次；截断的 5 次用 `--max-tokens 32768` 补跑 | 30 次里 26 次有结果；4 次 JSON 不合法，原文在 `content`，人工判 |
| `2026-11-16-walkthrough_first-flash-replay/` | 当前配置（开思考），只有 agent-window 两轮 | 2 轮 |

没有收进来的运行：

- 当前配置（开思考）的其余轮次：DeepSeek 余额用完（402），审查员全部失败却记为完成（issue #91）。
- 当前配置（关思考）的 5 轮：工具循环没回复参数不合法的工具调用，请求被拒（issue #96）。

## 接下来

1. issue #96（PR #97）、issue #91（PR #92）合入后，接入可观测性（issue #16，Langfuse）再跑。
2. 当前配置跑 mirror-tools、increment、model-stream、verify-schedule 各 3 轮，开、关思考各一组。
3. 按 D-041 判分：弱基线 3/3 命中的题降为必过（初看：agent-window、context-markup）。
4. 把 issue #96 加成回放题：PR #53 的 base 到合入时的 head，答案是 PR #97。
