# 黄金评测集

用来衡量 review 服务的 PR 题目：每道题是一个真实或构造的 PR，加上一份可以验证的答案。

格式的权威定义在 `src/review_eval/case.py`，每个字段的含义写在字段的 `description` 里。检查格式：

    uv run python -m review_eval check golden/cases --answers golden/cases

只给题目目录、不给 `--answers` 时，只检查题目，用于答案不在手边的情况，比如考试集。

## 目录

    golden/cases/<id>/
      case.yaml     review 拿到的输入：仓库、base 和 head、PR 描述、关联 issue、仓库配置
      answer.yaml   答案：应该找到什么，提了不算误报的点，来源，标签，复核人

两份文件分开放，是为了让答案碰不到 reviewer。开发集的答案和题目放在一起；考试集（`split: test`）的答案放在调 skill 的人碰不到的地方，用 `--answers` 指过去，按 `<id>/answer.yaml` 的同样结构存放。

## 规则

- **`split`**：`dev` 用来调 skill，失败可以随便看；`test` 只在定版时跑，不看失败细节；`smoke` 只检查流程能不能跑通。
- **PR 描述要写明使用场景。** 🔴 的定义是"在 PR 描述的场景里会发生"，没有场景就判不了严重程度。
- **issue 要冻结。** 写进 `case.yaml` 的是做题时 issue 的样子，线上的 issue 之后可能会改。
- **每条答案都要有证明（`proof`）**，三种：
  - `fail_to_pass`：把 `fix_commit` 里的 `test_paths` 拿到 head 上，跑 `command` 必须以 `head_exit_code` 退出；在 `fix_commit` 上跑必须以 0 退出。
  - `repro`：在 head 上跑 `command`，必须以 `head_exit_code` 退出。
  - `manual`：没有命令，`note` 里写清是谁、怎么确认的。
- **`head_exit_code` 必须对应失败本身。** pytest 测试失败退出 1、收集出错退出 2；Python 报错也是 1。所以修复提交里的测试文件如果在 head 上导入不了，就证明不了什么，这时改用 `repro`，并用一个报错不会产生的退出码，例如 3。样例题里的 `quote-across-blank-line` 就是这种情况。
- **`severities` 和 `dimensions` 是列表**：列出的都算对。答案不会是 nit：没人提的 nit 不算漏报。
- **`acceptable`**：真实但答案没要求的点，比如已存在的问题、合理的补测试建议。只要不超过 `max_severity`，提了就不算误报。
- **`verified`**：由人复核后填写 `by` 和 `on`。没填的是草稿，不计分。
- **不能离线检查的内容由评测脚本检查**：提交是否存在、行号是否在 head 的文件范围内、证明是否成立。

## 样例

`review-lab-basis-checks`：本仓库的 `db3a1d0`，issue #68 的第一版实现。它有三个后来在 review 里发现、各自被修复的 bug，三条证明都实际跑过：在 head 上失败，在修复提交上通过。严重程度和维度是初步判断，还没复核（`verified: null`）。
