# 给 Claude 的工作约定

## 代码审查

- 每次写完代码、提交之前，派一个子 agent 对本次改动运行 `/code-review high`，不需要用户每次提醒。
- 子 agent 的发现用 ReportFindings 报告；修完后再报一次，每条写明结果（fixed / skipped / no_change_needed）。
- 只改文档、不改代码时可以不跑，在 PR 的「验证」里说明。
