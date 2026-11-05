# UniCaMo review + HumanoidWM-4090: thiết kế diffusion world model cho Humanoid Robot Learning

Ngày: 2026-11-05. Nhãn: `[SRC]` nguồn nêu rõ · `[SELF]` số tác giả tự báo cáo · `[INF]` suy luận/ước tính của tôi, **chưa đo** · `[ ? ]` chưa kiểm.

## TIP

| Mục | Nội dung |
|---|---|
| Mục tiêu | (1) Đánh giá UniCaMo (arXiv 2607.02798v1) và cái gì chuyển được sang robot; (2) đề xuất kiến trúc diffusion world model cho humanoid, train được trên 1× RTX 4090 (24 GB) |
| Đầu ra | File này, đã push lên nhánh `claude/sleepy-shannon-duhe8f` |
| Input | Văn bản paper bạn dán (không tải lại từ arXiv) và kết quả WebSearch ngày 2026-11-05 |
| AC-01 | Review chỉ ra được các điểm không nhất quán, trích đúng số trong bảng |
| AC-02 | Kiến trúc có: backbone, conditioning, loss, bộ nhớ, kế hoạch train theo giai đoạn |
| AC-03 | Ước tính VRAM ghi rõ là ước tính + có kế hoạch đo |
| AC-04 | Tách phần đã có nguồn khỏi phần tôi suy luận |
| Ngoài phạm vi | Chưa viết code train, chưa đo VRAM thật |

## 1. Review UniCaMo

### Paper làm gì `[SRC: bản dán]`
- Không đổi kiến trúc: chỉ **dựng lại noise đầu vào** của video diffusion (Wan 2.1 I2V 14B + LoRA rank 64).
- *Track-the-Noise*: copy noise khung 1 theo quỹ đạo điểm 3D đã chiếu về latent (patch 3×3), xử lý va chạm bằng z-buffer (điểm gần camera thắng).
- *Move-the-World*: noise Gaussian dán trên mặt cầu bao quanh camera; mỗi khung lấy mẫu bằng ray-sphere intersection → vùng mới lộ ra khi camera di chuyển vẫn nhất quán giữa các khung.
- Ghép: ưu tiên noise từ tracks, chỗ trống lấy từ sphere; pha với Gaussian thuần (degradation rate θ=0.2) vì noise dựng lại không còn Gaussian theo thời gian.
- Train: 400K clip, 32×H100, 3 ngày `[SELF]`. Overhead warping 0.2 s/sample `[SELF]`.

### Điểm không nhất quán (đối chiếu số trong văn bản dán)

| # | Phát hiện | Vị trí |
|---|---|---|
| 1 | Text ghi "PSNR 19.28 vs 19.10" cho ablation sphere, nhưng Table 2(a) ghi **19.41** vs 19.10. Số 19.28 là của Table 3 (N=1024) | §4.3 vs Table 2/3 |
| 2 | Cùng split "MoveBench single-object": Table 1 PSNR **18.4**, Table 2/3 PSNR **19.2–19.4**. Không giải thích khác biệt (protocol, subset?) | Table 1 vs 2, 3 |
| 3 | Chọn θ=0.2 là "best balance", nhưng θ=0.0 bằng hoặc tốt hơn trên PSNR (19.46), EPE (2.19), Temporal Flickering (98.33), Motion Smoothness (99.13). Chênh lệch giữa θ=0.0 và 0.2 chỉ ~0.02–0.07 điểm | Table 2(b) |
| 4 | Gain của sphere noise nhỏ: +0.3 dB PSNR (nếu dùng 19.41), EPE 2.42→2.21; không có std/seed | Table 2(a) |
| 5 | Nói Go-with-the-Flow "không tạo video có ý nghĩa" nhưng Table 1 vẫn báo số cho nó | §4.2 vs Table 1 |
| 6 | Text ghi caption bằng Qwen2.5-VL, tài liệu [5] là Qwen3-VL | §3.3 vs refs |
| 7 | Ground truth 3D tracks và pose là **ước lượng** (TAPIP3D, ViPE) từ 2D tracks của MoveBench; EPE đo trên tracks ước lượng, nên bias có thể thiên về phương pháp dùng cùng bộ ước lượng | §4.1 |
| 8 | Không có metric riêng cho camera control (rotation/translation error); "joint control" chỉ chứng minh qua hình định tính | §4 |
| 9 | Robotics chỉ xuất hiện ở động cơ ([41], [25]); **không có thí nghiệm robot nào** | §1 |

Kết luận review `[INF]`: ý tưởng sạch, chi phí inference thấp, đáng mượn; nhưng độ mạnh bằng chứng ở mức "preprint, một benchmark, chưa tái lập". Tôi chưa đối chiếu với bản arXiv gốc, chỉ với bản bạn dán.

### Cái gì chuyển được sang humanoid `[INF]`
1. **Robot đã biết tracks sẵn**: forward kinematics (FK) + calibration cho vị trí 3D của các link/ngón tay/bàn chân trong mọi khung. Không cần TAPIP3D hay người kéo chuột. → *FK-Track-the-Noise*.
2. **Camera đầu (ego-view) di chuyển cùng thân**: pose camera lấy từ FK + IMU/VIO (đúng mảng bạn đang làm). Vùng lộ ra khi robot quay đầu/đi là đúng chỗ *Move-the-World* hữu ích.
3. Không thêm nhánh điều khiển → bớt VRAM và bớt tham số phải train, hợp 4090.

Rủi ro: (a) paper tự nhận noise dựng lại mất Gaussianity theo thời gian; (b) sphere proxy thô, phòng bệnh viện/phòng bếp có hình học nhiều lớp; (c) chưa ai chứng minh trên humanoid. Phải coi đây là **giả thuyết cần ablation**, không phải thành phần mặc định.

## 2. Bối cảnh các world model cho robot (từ search, chưa đọc toàn văn)

| Hướng | Ý chính | Nguồn |
|---|---|---|
| UWM | Một transformer chứa cả action diffusion lẫn video diffusion, mỗi modality có timestep riêng → cùng một model làm policy / forward dynamics / inverse dynamics / video generator; học được từ video không có action | [RSS 2025](https://roboticsproceedings.org/rss21/p015.html), [arXiv 2504.02792](https://arxiv.org/abs/2504.02792v1) |
| DreamGen | Fine-tune I2V model trên robot → sinh video → gán action bằng IDM/latent action → train policy; tác giả báo humanoid làm 22 động từ mới ở 10 môi trường mới `[SELF]` | [NVIDIA](https://research.nvidia.com/labs/gear/dreamgen), [arXiv 2505.12705](https://arxiv.org/abs/2505.12705) |
| World Action Model (DreamZero, DiT4DiT, Cosmos Policy, UVA) | Dự đoán đồng thời video tương lai và action; DreamZero 14B autoregressive chạy closed-loop 7 Hz `[SELF]` | [Survey 2605.00080](https://arxiv.org/pdf/2605.00080), [NVIDIA glossary](https://www.nvidia.com/en-us/glossary/world-action-model.md) |
| Vid2World | Biến video diffusion thụ động thành world model causal, action-conditioned, dùng diffusion forcing + action dropout | [alphaxiv](https://alphaxiv.org/overview/2505.14357v2) |
| Causal Forcing++ / Self Forcing / NFD | Chưng cất sang autoregressive few-step để real-time; NFD 310M đạt >30 FPS trên A100 `[SELF]` | [arXiv 2605.15141](https://arxiv.org/pdf/2605.15141), [arXiv 2506.01380](https://arxiv.org/html/2506.01380v2) |
| Wan2.1 T2V-1.3B | Inference cần 8.19 GB VRAM (số chính thức trên model card); 5 s 480p ≈ 4 phút trên 4090 | [HF card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) |

Giới hạn quan trọng: các nguồn **không** có công thức train world model đầy đủ trên một GPU. ForgeWM yêu cầu 8 GPU để tái lập. Phần distillation few-step nhiều khả năng **vượt 4090**; xem §5.

## 3. Kiến trúc đề xuất: HumanoidWM-4090 `[INF]`

```
 ego/third-person video ──► Frozen video VAE ──► cached latents z (offline)
 proprio q_t (joints), IMU ─┐
 action chunk a_{t:t+H} ────┼──► tokenizers ──┐
 camera pose (FK+VIO) ──────┘                 ▼
 FK keypoint tracks ──► Track-the-Noise  ──► structured source noise ε̃  (optional, ablate)
 camera path ─────────► Move-the-World   ──┘
                                              ▼
              Causal Video-Action DiT (flow matching, per-modality timesteps)
                 ├─ video tokens  (chunk-causal attention, KV cache)
                 └─ action tokens (small head, bidirectional inside chunk)
                                              ▼
   modes: policy | forward dynamics (world model) | inverse dynamics | action-free video
```

### Quyết định kiến trúc và lý do

| Quyết định | Chọn | Vì sao (why) |
|---|---|---|
| Latent space | **VAE đóng băng, cache latent offline** | Train không bao giờ nạp VAE → tiết kiệm VRAM và thời gian; đây là đòn bẩy lớn nhất trên 24 GB |
| Backbone | **Wan2.1-1.3B + LoRA** (Phương án A, khuyến nghị) hoặc DiT 300–500M train từ đầu (B) | A tận dụng prior video; 1.3B có inference 8.19 GB `[SRC]`. B nếu domain (robot) quá khác hoặc A hết VRAM. 14B như paper là ngoài tầm 4090 |
| Objective | **Flow matching / rectified flow** (như Wan và π0) | Ít bước hơn DDPM, ổn định; cùng họ với backbone nên dễ fine-tune |
| Rollout | **Chunk-causal + KV-cache**, train bằng diffusion forcing (noise level riêng từng khung) | World model phải rollout dài theo action; bidirectional không dùng được online |
| Action | Token action nhỏ, **timestep riêng** như UWM | Một model, nhiều chế độ; dùng được video không nhãn action (nhiều hơn rất nhiều dữ liệu có nhãn) |
| Conditioning | AdaLN/token cho action + proprio + camera pose; **không** ControlNet | Tiết kiệm VRAM; khớp tinh thần UniCaMo |
| Structured noise | FK-Track + sphere noise làm **source noise**, tuỳ chọn, mặc định OFF | Chưa chứng minh trên humanoid; phải ablation (§6) |
| Action dropout | Có (như Vid2World) | Cho phép CFG theo action và học chế độ không điều kiện |

### Ngân sách tính toán / VRAM (ước tính, **phải đo**)

| Mục | Phương án A (1.3B + LoRA r=64) | Ghi chú |
|---|---|---|
| Weights bf16 | ~2.6 GB | 1.3B × 2 byte |
| LoRA params + optimizer (8-bit Adam) | < 1 GB | rất nhỏ so với weights |
| Activations | phụ thuộc token; dùng gradient checkpointing + FlashAttention | ví dụ 256 px: latent 32×32, patch 2×2 → 256 token/khung; 9 khung latent ≈ 2.3K token `[INF: giả định VAE 8×8×4 và patch 2×2, chưa kiểm]` |
| Ước tính tổng | ~8–16 GB, vừa 24 GB | Full fine-tune cần ~8 GB cho gradient + optimizer và có thể phải bỏ fp32 master weights; borderline, không khuyến nghị bắt đầu từ đây |

Kế hoạch đo: chạy 20 bước với batch 1, ghi `torch.cuda.max_memory_allocated()` theo (độ phân giải, số khung, rank LoRA); chọn cấu hình nằm dưới ~20 GB để chừa headroom.

## 4. Kế hoạch train 4 giai đoạn

| Giai đoạn | Việc | Dữ liệu | Đạt khi (AC) |
|---|---|---|---|
| S0 | Encode toàn bộ video → latent cache; chuẩn hoá FK, pose, action | Dữ liệu humanoid của bạn + video egocentric không nhãn | Đọc lại latent decode ra video, PSNR/SSIM khớp mong đợi của VAE |
| S1 | Thích nghi domain (action-free) bằng LoRA | Video robot + egocentric | FVD trên tập giữ lại giảm so với backbone gốc |
| S2 | Action-conditioned + diffusion forcing + action dropout | Video có action | Forward model: sai số keypoint (EPE) giữa video sinh và FK chiếu thấp hơn baseline không điều kiện |
| S3 | Thêm action head (policy/IDM), timestep riêng | Có nhãn action | Policy chạy được trong sim hoặc replay, IDM gán nhãn đúng trên tập kiểm |
| S4 (tuỳ chọn) | Few-step distillation cho real-time | — | **Có thể cần >1 GPU**; làm sau khi S1–S3 đạt |

Bộ chỉ số đánh giá: FVD/FID/PSNR/SSIM (như paper), EPE keypoint (mượn từ UniCaMo), độ trôi khi rollout dài (error theo bước), và **hiệu quả downstream** (policy huấn luyện trên dữ liệu sinh có tốt hơn không). Giữ tập kiểm hoàn toàn dữ liệu thật.

## 5. Triển khai và an toàn (nối với dự án AMR/Jetson của bạn)

- World model **không** chạy ở tốc độ điều khiển trên Jetson Orin Nano: số liệu đã thu thập cho Stable Diffusion trên Orin Nano là ~49–139 s/ảnh `[SELF]` ([arXiv 2407.19547](https://arxiv.org/pdf/2407.19547)). Dùng world model trên máy 4090 để: sinh dữ liệu (kiểu DreamGen), đánh giá policy offline, lập kế hoạch ngoại tuyến.
- Online chỉ giữ policy nhỏ đã distill (kiểu UVA bỏ qua bước sinh video) `[INF]`.
- Mọi đầu ra từ world model/policy học được phải đi qua lớp an toàn cổ điển (giới hạn khớp, NMPC/ZMP, watchdog trên STM32). World model có thể "hallucinate"; không dùng làm bằng chứng an toàn.

## 6. Thí nghiệm ablation bắt buộc trước khi tin structured noise

1. Gaussian thuần vs FK-Track noise vs FK-Track + sphere (cùng seed, ≥3 seed, báo std).
2. θ ∈ {0, 0.2, 0.5} (paper cho thấy chênh lệch rất nhỏ, cần xem có đáng không).
3. Ego-view có camera chuyển động vs camera cố định (sphere noise chỉ nên giúp ở trường hợp đầu).
4. Tác động lên rollout dài (độ trôi), không chỉ clip ngắn.

## Báo cáo bàn giao

**Trạng thái: PARTIAL**

| AC | Kết quả | Evidence |
|---|---|---|
| AC-01 | PASS | 9 điểm ở §1 đều trích số từ bản dán; tự đối chiếu, chưa so với PDF gốc |
| AC-02 | PASS | §3–§4 |
| AC-03 | PASS (có điều kiện) | VRAM ghi `[INF]`, kèm kế hoạch đo; chưa chạy |
| AC-04 | PASS | Nhãn trong từng bảng |

- **Kiểm đã chạy:** tự kiểm (đọc đối chiếu bảng/văn bản). Không phải kiểm độc lập.
- **Chưa làm:** không tải lại paper gốc; không đọc toàn văn UWM/Vid2World/DreamZero (chỉ snippet); không đo VRAM; không viết code.
- **Giả định chưa kiểm:** cấu hình VAE/patch của Wan 2.1 (8×8×4, patch 2×2); khả năng full fine-tune 1.3B trên 24 GB.
- **Cần chủ việc quyết:** (1) robot humanoid cụ thể (Unitree G1? số DoF, camera?) và dữ liệu hiện có; (2) mục tiêu chính của world model: sinh dữ liệu, đánh giá policy, hay planning; (3) ưu tiên Phương án A (1.3B + LoRA) hay B (DiT nhỏ từ đầu).
