# Scenario Factory: app sinh dữ liệu cho Humanoid bằng diffusion / world model

Ngày: 2026-11-05. Nhãn: `[SRC]` nguồn nêu rõ · `[SELF]` số tác giả/vendor tự báo cáo · `[INF]` suy luận hoặc ước tính của tôi, chưa đo · `[ ? ]` chưa kiểm.
Toàn bộ mục `[SRC]` lấy từ kết quả WebSearch (tóm tắt), **chưa đọc toàn văn**; nhiều nguồn là preprint chưa tái lập.

## TIP

| Mục | Nội dung |
|---|---|
| Mục tiêu | Chốt kiến trúc MVP cho app công ty: sinh nhiều tình huống mà dữ liệu thật không thu được (hiếm, nguy hiểm, tốn kém), dùng làm dữ liệu train/đánh giá cho humanoid |
| Đầu ra | File này (nối tiếp `2026-11-05-humanoid-world-model-design.md`) |
| AC-01 | Bảng so sánh ≥12 model/hệ thống, mỗi dòng có nguồn và vai trò trong stack |
| AC-02 | Kiến trúc có pipeline, module, cơ chế lọc chất lượng, provenance |
| AC-03 | MVP có phạm vi, tiêu chí đo được, ngân sách GPU |
| AC-04 | Tách nguồn / tự báo cáo / suy luận |

## 1. Khảo sát: các họ model và vai trò

| Họ | Model / hệ thống | Làm gì | Nguồn | Vai trò đề xuất |
|---|---|---|---|---|
| Image/video nền | Latent Diffusion (SD) | Diffusion trong latent của autoencoder | [arXiv 2112.10752](https://arxiv.org/abs/2112.10752) | Nền tảng; tăng dữ liệu ảnh |
| | Wan 2.1 (1.3B / 14B) | Video diffusion mã nguồn mở; 1.3B cần 8.19 GB, ~4 phút cho 5 s 480p trên 4090 | [HF card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) | Backbone fine-tune trên 4090 |
| | LTX-Video | Video latent diffusion nhanh: 5 s 24 fps 768×512 trong ~2 s trên H100 `[SELF]`; license các bản sau chưa rõ | [arXiv 2501.00103](https://arxiv.org/abs/2501.00103v1) | Ứng viên sinh nhanh; `[ ? ]` license |
| | HunyuanVideo (13B+) | Chất lượng cao, nặng; có bản distill 6 bước | [so sánh](https://www.whitefiber.com/blog/best-open-source-video-generasi-model) | Quá nặng cho 4090 khi train |
| World foundation | NVIDIA Cosmos Predict / Transfer / Reason | Predict: dự đoán tương lai; **Transfer: biến video từ sim thành photoreal, đổi ánh sáng/môi trường**; Reason: VLM hiểu video; open weights, license cho dùng thương mại có điều kiện guardrail | [NVIDIA](https://www.nvidia.com/en-eu/ai/cosmos), [GitHub Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5), [arXiv 2511.00062](https://arxiv.org/pdf/2511.00062) | Lõi appearance + judge; `[ ? ]` kiểm license và version hiện hành |
| | Genie 2 / Genie 3 | World model tương tác, 24 fps 720p, promptable events; **chưa public** | [Engadget](https://www.engadget.com/ai/google-deepminds-genie-3-can-dynamically-alter-the-state-of-its-simulated-worlds-140052124.html) | Chỉ tham khảo, không dùng được |
| | V-JEPA 2 / 2-AC | Dự đoán trong latent (không sinh pixel); predictor ~300M, MPC với CEM, 62 h dữ liệu Droid, zero-shot trên Franka | [arXiv 2506.09985](https://arxiv.org/html/2506.09985v1) | Hướng planning nhẹ, ít hallucinate pixel |
| | Dreamer 4 | RL hoàn toàn trong world model, offline; causal tokenizer + block-causal transformer; real-time 1 GPU | [arXiv 2509.24527](https://arxiv.org/abs/2509.24527?context=cs) | Mẫu kiến trúc cho world model nhỏ, RL trong tưởng tượng |
| Robot policy / WAM | Diffusion Policy | Visuomotor policy bằng action diffusion | [RSS](https://roboticsproceedings.org/rss19/p026.html) | Policy baseline |
| | π0 | VLA + flow matching action expert | [arXiv 2410.24164](https://arxiv.org/html/2410.24164) | Policy mục tiêu cho dữ liệu sinh |
| | UWM | Video + action diffusion trong 1 transformer, timestep riêng; học từ video không nhãn | [RSS 2025](https://roboticsproceedings.org/rss21/p015.html) | **Backbone world-model + policy của ta** |
| | DreamGen / GR00T-Dreams | Fine-tune I2V trên robot → sinh video → gán action bằng IDM/latent action → train policy; humanoid 22 verb mới `[SELF]` | [NVIDIA](https://research.nvidia.com/labs/gear/dreamgen), [arXiv 2505.12705](https://arxiv.org/abs/2505.12705) | **Mẫu pipeline dữ liệu chính** |
| | Ctrl-World | World model nhiều camera, policy-in-the-loop, xếp hạng policy; +44.7% success sau SFT bằng quỹ đạo sinh `[SELF]` | [arXiv 2510.10125](https://arxiv.org/html/2510.10125v1) | Mẫu evaluator + data generator |
| | Scalable Policy Evaluation / WorldEval | Video model làm simulator, VLM chấm thành công, tương quan với thực tế; nêu rõ lỗi với action ngoài phân phối và vật lý kém | [arXiv 2511.11520](https://arxiv.org/html/2511.11520v3) | Căn cứ cho cổng chất lượng và cảnh báo |
| Causal / real-time | Vid2World, Causal Forcing++, Self Forcing, NFD | Chuyển video model thành causal, action-conditioned, few-step | [Vid2World](https://alphaxiv.org/overview/2505.14357v2), [arXiv 2605.15141](https://arxiv.org/pdf/2605.15141), [NFD](https://arxiv.org/html/2506.01380v2) | Giai đoạn tối ưu sau MVP |
| Chuyển động humanoid | DreamControl-v2, OMG-DiT (1000+ h G1), ECHO, RoboForge | Diffusion sinh motion trực tiếp trong không gian robot (G1), kèm tracker thực thi | [2604.00202](https://arxiv.org/pdf/2604.00202), [2606.10340](https://arxiv.org/pdf/2606.10340.pdf), [2603.16188](https://arxiv.org/pdf/2603.16188.pdf), [2603.17927](https://www.alphaxiv.org/abs/2603.17927) | Sinh hành vi con người/robot đa dạng, có thể thực thi |
| Cảnh 3D | World Labs Marble → Gaussian splat + collider → Isaac Sim (NuRec) | Text/ảnh → cảnh 3D có vật lý | [NVIDIA blog](https://developer.nvidia.com/blog/simulate-robotic-environments-faster-with-nvidia-isaac-sim-and-world-labs-marble), [Isaac Sim 6.0](https://radiancefields.com/nvidia-s-isaac-sim-6.0-ships-with-nurec-gaussian-splatting) | Môi trường đa dạng, có vật lý thật |
| | GaussGym | Gaussian splatting + physics GPU | [tóm tắt](https://www.emergentmind.com/topics/gaussgym) | Phương án mã nguồn mở thay Isaac |
| Điều hướng | NoMaD | Diffusion policy cho navigation/exploration | [Project](https://general-navigation-models.github.io/nomad) | Nếu humanoid cần đi lại |
| Điều khiển noise | UniCaMo | Điều khiển vật thể + camera qua noise đầu vào, không thêm nhánh | arXiv 2607.02798 (bản bạn dán) | Tuỳ chọn ở lớp video, mặc định tắt |

Quan sát chéo `[INF]` rút từ bảng:
- Hầu hết nguồn dùng world model làm **evaluator** hoặc **data generator có lọc**, không ai chứng minh nó thay hoàn toàn dữ liệu thật.
- Các nguồn đều nêu rủi ro: action ngoài phân phối và vật lý sai ([Scalable Policy Evaluation](https://arxiv.org/html/2511.11520v3)). Vì vậy kiến trúc dưới đây không để video model một mình "quyết" tình huống.

## 2. Nguyên tắc thiết kế (why)

1. **Physics là sự thật, diffusion là vẻ ngoài và sự đa dạng.** Sim cho action, va chạm và nhãn chính xác; diffusion thêm photorealism và biến thiên (đây là chính xác cách Cosmos Transfer được mô tả: biến video từ sim thành photoreal `[SRC]`). Video model thuần tuý chỉ dùng cho tình huống mở khi sim không dựng được, và phải qua cổng chất lượng chặt hơn.
2. **Mọi mẫu sinh ra phải có "hợp đồng" nhãn**: action từ sim hoặc IDM, kèm điểm tin cậy; mẫu thiếu nhãn tin cậy thì chỉ dùng cho pretrain không nhãn (kiểu UWM).
3. **Tập kiểm toàn dữ liệu thật**, tách hẳn. Dữ liệu sinh không bao giờ vào test.
4. **Provenance bắt buộc** (generator, version, seed, prompt, ScenarioSpec, điểm judge). Dùng cùng tinh thần Refinery của bộ kit: truy nguồn được.
5. **Judge độc lập với generator** để tránh tự chấm điểm cao cho chính mình.

## 3. Kiến trúc: Scenario Factory

```
                    ┌────────────────────────────────────────────┐
  Operator (web UI) │  1. Scenario Studio                         │
  mô tả bằng văn bản│  text/ảnh → ScenarioSpec (JSON, có schema)  │
  hoặc template     │  LLM/VLM + validator + thư viện hazard     │
                    └───────────────┬────────────────────────────┘
                                    ▼
            ┌──────────── 2. Scene & Motion Layer ─────────────┐
            │ Scene: Marble/asset → splat+collider → Isaac Sim │
            │ Motion: text→motion diffusion (robot space)      │
            │         + whole-body tracker → physically valid  │
            │ Output: video sim + action + proprio + masks     │
            └───────────────┬──────────────────────────────────┘
                            ▼
            ┌──────────── 3. Appearance Layer ─────────────────┐
            │ Cosmos-Transfer-style: ánh sáng, vật liệu, thời  │
            │ tiết, distractor; (tuỳ chọn) noise có điều khiển │
            └───────────────┬──────────────────────────────────┘
                            ▼
            ┌──────────── 4. Open-ended World Model ───────────┐
            │ HumanoidWM (UWM-style, Wan 1.3B + LoRA):         │
            │  • what-if từ ảnh thật (người ngã, vật rơi…)     │
            │  • forward dynamics / IDM gán action            │
            └───────────────┬──────────────────────────────────┘
                            ▼
            ┌──────────── 5. Quality Gate ─────────────────────┐
            │ a. Kinematic: FK keypoints vs track trên video   │
            │ b. VLM judge (task/hazard đúng spec?)            │
            │ c. Artifact/temporal check, dedup, diversity     │
            │ d. Human audit mẫu ngẫu nhiên                    │
            │ → ACCEPT / QUARANTINE / REJECT + lý do           │
            └───────────────┬──────────────────────────────────┘
                            ▼
   6. Dataset Registry (provenance, version) ──► 7. Train (UWM/π0/DiffPolicy)
                                                  ──► 8. Eval: world-model rollouts
                                                       + tập thật nhỏ (nguồn sự thật)
```

### Module và công nghệ gợi ý `[INF]`

| Module | Công nghệ | Ghi chú |
|---|---|---|
| Frontend | Web app (React) | Scenario builder, thư viện, hàng đợi review, so sánh A/B |
| API & queue | FastAPI + Redis/Celery; GPU workers tách theo loại job | Job idempotent, có seed |
| Lưu trữ | Postgres (metadata/provenance) + object store (video, latent) | Latent cache để train nhanh |
| Sim | Isaac Sim/Lab (hoặc GaussGym) | Tạo action/nhãn chính xác |
| Video | Wan 1.3B (LoRA), Cosmos (open weights) | `[ ? ]` kiểm license thương mại từng model |
| Judge | VLM (ví dụ Cosmos Reason) + kiểm tra hình học | Chạy trên model khác với generator |
| Export | Định dạng LeRobot-compatible `[ ? ]` | Tương thích π0 qua LeRobot port |

### Cổng chất lượng (cốt lõi của sản phẩm)

| Kiểm | Cách đo | Hành động |
|---|---|---|
| Kinematic consistency | Chiếu FK keypoints vào ảnh, so với tracker trên video sinh; ngưỡng EPE `[proposal: đặt sau khi đo baseline]` | Fail → REJECT |
| Spec adherence | VLM trả lời bộ câu hỏi sinh tự động từ ScenarioSpec ("có người băng qua?") | Fail → QUARANTINE |
| Physical plausibility | Kiểm xuyên vật, vật bay, đổi danh tính (VLM + heuristic) | Fail → REJECT |
| Diversity | Embedding clustering; giới hạn số mẫu/cụm | Cắt bớt cụm trùng |
| Label confidence | Sim = cao; IDM = đo độ phù hợp trên tập có nhãn thật | Thấp → chỉ dùng không nhãn |
| Human audit | Lấy mẫu ngẫu nhiên, đo tỷ lệ đồng ý với tự động | Hiệu chỉnh ngưỡng |

## 4. MVP

**Phạm vi (đề xuất, chủ việc chốt kịch bản cụ thể):** 3 họ tình huống khó thu thật, mỗi họ ≥ 5 biến thể tham số:
1. Người đột ngột xuất hiện/cắt ngang đường đi.
2. Vật rơi hoặc trượt khỏi tay robot.
3. Điều kiện nhìn kém (thiếu sáng, phản chiếu, khói/hơi nước).

**Đường chính (sim-grounded):** Spec → Isaac → Appearance → Gate → dataset.
**Đường phụ (open-ended):** ảnh thật → HumanoidWM what-if → IDM → Gate (chặt hơn). Chỉ bật sau khi đường chính ổn.

### Tiêu chí MVP (AC-MVP, ngưỡng là đề xuất cần chủ việc duyệt)

| ID | Tiêu chí | Cách kiểm |
|---|---|---|
| M-01 | Từ 1 mô tả văn bản tạo ra ScenarioSpec hợp lệ schema ≥ 95% trong 100 mô tả thử | Validator tự động |
| M-02 | Pipeline chạy end-to-end không can thiệp tay cho 3 họ tình huống | Chạy 1 lượt, log |
| M-03 | Mỗi mẫu ACCEPT có đủ provenance + nhãn action | Kiểm schema trên toàn bộ |
| M-04 | Tỷ lệ đồng ý giữa gate tự động và người trên 200 mẫu audit đạt mức do chủ việc đặt | Đối chiếu nhãn người |
| M-05 | Policy train bằng thật+sinh không tệ hơn policy chỉ thật trên **tập kiểm thật** của các tình huống tương ứng | Benchmark thật, ≥3 seed |
| M-06 | Mọi đầu ra gắn cờ synthetic, không lẫn vào test | Kiểm tự động trước export |

Ngân sách GPU `[INF]` từ số đo đã nêu (Wan 1.3B, 5 s 480p ≈ 4 phút trên 4090, chưa tối ưu `[SRC]`): ≈360 clip/ngày/GPU → 10.000 clip ≈ 28 GPU-ngày. Muốn nhanh hơn: giảm độ phân giải, rút ngắn clip, chưng cất few-step (có thể cần nhiều GPU hơn 4090), hoặc dùng cloud cho sinh hàng loạt và giữ 4090 cho fine-tune/dev.

Lộ trình tham khảo `[INF]`: tuần 1–2 Spec + sim + export; tuần 3 appearance + gate; tuần 4 train/eval nhỏ + audit; sau đó mới bật đường open-ended.

## 5. Rủi ro và giảm thiểu

| Rủi ro | Giảm thiểu |
|---|---|
| Video hallucinate vật lý | Sim làm sự thật; gate kinematic; đường open-ended bị giới hạn |
| Sim-to-real gap | Appearance diversification; luôn có tập kiểm thật; domain randomization |
| Policy "khai thác" world model (reward hacking) | Đánh giá cuối bằng dữ liệu thật; judge khác generator |
| Nhãn IDM sai | Đo trên tập có nhãn thật; dùng điểm tin cậy; hạ cấp thành dữ liệu không nhãn |
| Mode collapse / thiếu đa dạng | Giới hạn theo cụm embedding; tham số hoá Spec |
| License thương mại | Kiểm Cosmos/Wan/LTX trước khi đưa vào sản phẩm (`[ ? ]`) |
| An toàn robot thật | Dữ liệu sinh không phải bằng chứng an toàn; giữ lớp an toàn cổ điển và thử nghiệm có kiểm soát |

## Báo cáo bàn giao

**Trạng thái: PARTIAL**

| AC | Kết quả | Evidence |
|---|---|---|
| AC-01 | PASS (mức tóm tắt) | Bảng §1: ~20 hệ thống, mỗi dòng có link; nhiều dòng chỉ từ snippet search |
| AC-02 | PASS | §2–§3 |
| AC-03 | PASS (đề xuất) | §4; ngưỡng chưa đặt, ngân sách là ước tính |
| AC-04 | PASS | Nhãn trong bảng |

- **Kiểm đã chạy:** tự kiểm (đối chiếu giữa các kết quả search). Không phải kiểm độc lập.
- **Chưa kiểm:** license thương mại của Cosmos/Wan/LTX; phiên bản Cosmos hiện hành (nguồn nhắc cả Predict2.5 và "Cosmos 3"); khả năng chạy Cosmos Transfer trên 4090; định dạng export LeRobot; chưa có prototype nào.
- **Cần chủ việc quyết:** (1) humanoid cụ thể và tập tình huống ưu tiên; (2) có Isaac Sim/GPU cloud không, hay chỉ 4090; (3) yêu cầu license thương mại; (4) ngưỡng chấp nhận M-04/M-05.
