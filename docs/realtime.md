# 实时转写

实时和离线共用一份 R2T2 权重，每个连接独立保存音频、解码状态和说话人缓存。GPU 使用 vLLM，CPU 使用 Rust；启动与资源配置见[部署说明](deployment.md)。

## WebSocket 协议

连接 `ws://host:17003/v1/stream`。启用鉴权时使用 `Authorization: Bearer <API_KEY>`，浏览器可使用 `?token=...`。首条消息发送 `{}`，或带热词上下文：

```json
{"context":"网易有道，Qwen，R2T2"}
```

收到 `ready:true` 后发送 **16 kHz、单声道、int16 little-endian PCM** 二进制帧。建议每帧 160 ms（5120 字节），单帧最多 1 秒。结束时发送文本 `end`，继续接收至 `done:true`。不接受语言或采样参数，模型自动识别语言。

```json
{"delta":"Hello","audio_ms":960,"inference_ms":45.2,"done":false,"utterance":0,"utterance_end":false}
{"delta":".","audio_ms":1280,"inference_ms":48.1,"done":false,"utterance":0,"utterance_end":true}
{"utterance":0,"speaker":"说话人1"}
{"delta":"","audio_ms":1440,"inference_ms":3.0,"done":true,"utterance":1,"utterance_end":true,"text":"Hello."}
```

- 直接追加 `delta`，已发布文字不改写；最终 `text` 是完整文本。
- `audio_ms` 是已处理音频位置，不是文字时间戳；空增量可作为处理进度。
- `inference_ms` 包含该请求的调度和推理时间，不是端到端延迟。
- `utterance` 为停顿分隔的语句编号，`utterance_end` 表示该语句结束。

错误返回 `{"code":"capacity_exceeded","error":"..."}` 并关闭连接。每会话最长 1 小时，输入队列最多 10 秒；超时、积压或断线会取消请求并释放名额。

## 停顿自动结束

网关用 [Silero VAD](https://github.com/snakers4/silero-vad) 检测说话，累计至少 256 ms 语音后，若连续静音达到设定时长，就像客户端发送 `end` 一样收尾。服务端默认值由 `REALTIME_AUTO_STOP_MS` 设置（默认 `0`，即关闭），单个会话可在首条消息覆盖，`0` 表示关闭，最大 60000：

```json
{"context":"网易有道","auto_stop_silence_ms":3000}
```

`ready` 消息中的 `auto_stop_silence_ms` 是本会话实际生效值；VAD 加载失败时为 `0`，转写照常进行。触发后先收到

```json
{"auto_stop":"silence","audio_ms":5120}
```

随后照常收到补齐的文字、说话人标签和 `done:true`。客户端收到 `auto_stop` 后应停止发送音频；此后到达的音频帧和 `end` 会被忽略。静音按已接收音频计时，不按墙钟时间；客户端完全停止发送时仍由 30 秒接收超时关闭连接。持续背景人声或音乐可能被判为语音而推迟结束。OpenAI 兼容接口 `/v1/realtime` 不受此设置影响。

## 说话人标签

Nemotron 使用 1.04 秒缓冲模式，覆盖到语句尾部后发送该语句的主讲者标签。每个有文字的语句一个标签，通常比文字晚约 1–2 秒；全部标签先于 `done` 发送。`speaker:null` 表示无法判断或分离失败，文字转写继续。

编号仅在本连接内有效，没有词级时间戳。无停顿的抢话会归到同一语句的主讲者，背景噪声可能推迟停顿检测和标签。保存录音后，以离线重新识别和说话人分配结果为准。

## 延迟与限制

GPU 和 CPU 默认均每 640 ms 解码，可用 `R2T2_CHUNK_SECONDS` 调整，允许范围为 0.08–2 秒；输入帧长度与解码间隔独立。音频窗口最多 16 秒，超过后移除最早 8 秒及相应文本前缀。语音之后至少 320 ms 的低能量音频触发语句收尾，结束连接前也会补齐未确认文字。

GPU 默认支持 4 个会话，CPU 默认 1 个。GPU 离线最多并行识别 8 段，CPU 串行执行且不可中途抢占。共享推理、对齐和说话人模型会竞争资源，需按实际并发负载验收。首个解码窗口长度不等于首字延迟保证，混合语言、口音和重叠发言仍可能识别不完整。

## 验证

```bash
uv run --no-sync python -m scripts.benchmark.realtime_smoke zh.wav en.wav \
  --url http://127.0.0.1:17003 --concurrency 4 \
  --output benchmark_results/r2t2-acceptance.json
```

CPU 将并发改为 1，鉴权通过 `API_KEY` 环境变量提供。脚本覆盖真实节奏回放、语言切换、长音频、满载拒绝和断线回收；它检查接口与会话隔离，不代替人工准确率评测。

解码算法源自 [NetEase Youdao R2T2](https://github.com/netease-youdao/Confucius4-R2T2)，源码归属与许可见 `deploy/R2T2-NOTICE`。
