# Diffusion Models: Image Generation → Robot Learning (khảo sát có nguồn)

Ngày thu thập: 2026-11-05. Phương pháp: AI Work Method 2.0 (TIP → thu thập → kiểm → báo cáo).
**Giới hạn quan trọng:** mọi mục dưới đây lấy từ *kết quả WebSearch (bản tóm tắt)*, chưa mở và đọc toàn văn từng paper.
Nhãn: `[SRC]` = nguồn nêu rõ; `[SELF]` = số do tác giả tự báo cáo (chưa tái lập); `[INF]` = suy luận của tôi; `[ ? ]` = chưa kiểm.

## TIP

| Mục | Nội dung |
|---|---|
| Mục tiêu | Nắm bức tranh diffusion: lý thuyết → sinh ảnh → robot learning, và áp dụng cho AMR bệnh viện |
| Đầu ra | File này |
| AC-01 | Mỗi claim có URL nguồn |
| AC-02 | Đủ 3 mảng: lý thuyết, sinh ảnh, robot |
| AC-03 | Chỗ chưa kiểm hoặc mâu thuẫn được gắn nhãn |
| AC-04 | Có mục áp dụng cho Jetson Orin Nano + NMPC |

## 1. Nền tảng lý thuyết

| Chủ đề | Claim | Nhãn | Nguồn |
|---|---|---|---|
| DDPM (Ho et al., NeurIPS 2020) | Sinh ảnh bằng cách học đảo ngược quá trình thêm nhiễu Gaussian nhiều bước | SRC (thứ cấp) | [Overview 2412.01371](https://arxiv.org/pdf/2412.01371) |
| DDIM (Song, Meng, Ermon) | Giữ nguyên cách train của DDPM nhưng lấy mẫu nhanh hơn (implicit) | SRC | [HF Diffusers DDIM](https://huggingface.co/docs/diffusers/v0.35.1/api/schedulers/ddim) |
| Classifier-free guidance (Ho & Salimans) | Train chung conditional + unconditional, trộn score để đổi chất lượng lấy diversity | SRC (thứ cấp) | [Blog tổng hợp](https://betterprogramming.pub/diffusion-models-ddpms-ddims-and-classifier-free-guidance-e07b297b2869) |
| Ngày CFG | **Mâu thuẫn:** một nguồn ghi 26/7/2022, slide khác ghi preprint 2021 | `[ ? ]` | Kiểm tại arXiv 2207.12598 |

## 2. Sinh ảnh

| Chủ đề | Claim | Nhãn | Nguồn |
|---|---|---|---|
| Latent Diffusion (Rombach et al., CVPR 2022) | Chạy diffusion trong latent của autoencoder đã pretrain, giảm mạnh chi phí so với pixel-space; cross-attention cho text/bbox | SRC | [arXiv 2112.10752](https://arxiv.org/abs/2112.10752), [CVPR](https://openaccess.thecvf.com/content/CVPR2022/html/Rombach_High-Resolution_Image_Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.html) |
| Stable Diffusion = LDM | Quan hệ này là kiến thức nền của tôi, chưa kiểm trong lần tìm này | INF / `[ ? ]` | — |

## 3. Robot learning

| Hướng | Claim | Nhãn | Nguồn |
|---|---|---|---|
| **Diffusion Policy** (Chi et al., RSS 2023) | Mô hình hoá visuomotor policy như conditional denoising; thêm receding horizon, visual conditioning, time-series diffusion transformer; trung bình +46.9% trên 12 task / 4 benchmark | SELF | [RSS](https://roboticsproceedings.org/rss19/p026.html), [arXiv 2303.04137](https://arxiv.org/abs/2303.04137) |
| **Diffuser** (Janner et al., ICML 2022) | Diffuse toàn bộ trajectory cùng lúc; guide khi lấy mẫu để thoả ràng buộc/return; goal-conditioning như inpainting | SRC | [arXiv 2205.09991](https://arxiv.org/abs/2205.09991) |
| **NoMaD** (Sridhar et al., 2023) | Một diffusion policy cho cả goal-directed navigation và exploration nhờ goal masking; ViNT encoder + planner topological graph | SRC | [Project](https://general-navigation-models.github.io/nomad), [arXiv 2310.07896](https://arxiv.org/html/2310.07896v1) |
| NoMaD hiệu năng | Hơn ViNT >25% (exploration), "15× hiệu quả tính toán" | SELF | như trên |
| **π0** (Physical Intelligence) | VLM (PaliGemma) + action expert dùng flow matching, sinh action chunk liên tục, đến ~50 Hz | SRC (thứ cấp) | [arXiv 2410.24164](https://arxiv.org/html/2410.24164) |
| **Data augmentation bằng text-to-image** | ROSIE (Imagen Editor inpaint vật thể/nền/distractor), GenAug (depth-guided), CACTI (Stable Diffusion) | SRC (thứ cấp) | [ROSIE 2302.11550](https://arxiv.org/abs/2302.11550), [The Decoder](https://the-decoder.com/?p=3218) |
| Survey manipulation | Wolf et al.: grasp, trajectory planning, data augmentation; điểm yếu chính là sampling lặp chậm | SRC | [arXiv 2504.08438](https://arxiv.org/pdf/2504.08438), [Frontiers/PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12454101/) |
| Danh mục curated | Awesome-Robotics-Diffusion (có cả navigation, planning); survey "Diffusion Models in Robotics" chưa tìm thấy | `[ ? ]` | [list](https://www.sourcepulse.org/projects/1867844) |

## 4. Tăng tốc cho real-time

| Kỹ thuật | Claim | Nhãn | Nguồn |
|---|---|---|---|
| Vấn đề gốc | Diffusion policy cần 50–100 lần forward/action | SELF (preprint 2026) | [arXiv 2605.08511](https://arxiv.org/pdf/2605.08511.pdf) |
| Flow matching / rectified flow | Ít bước hơn; π0 vẫn dùng 10 bước tích phân | SRC | [SeFA-Policy 2511.08583](https://arxiv.org/html/2511.08583v1) |
| Consistency flow (FlowPolicy) | Inference 1 bước trên point cloud 3D | SELF | [arXiv 2412.04987](https://arxiv.org/html/2412.04987v2) |
| Consistency Policy (distillation) | 1.5 Hz → 62 Hz (~40×) | SELF | tổng hợp trong kết quả tìm; paper gốc chưa mở `[ ? ]` |
| Đánh đổi | Sau distill, action có thể lệch khỏi ground truth; các nguồn bất đồng về mức mất độ chính xác | Mâu thuẫn | SeFA-Policy ở trên |

## 5. Edge / Jetson

| Claim | Nhãn | Nguồn |
|---|---|---|
| Stable Diffusion trên Orin Nano: ~138.7 s (full precision) → ~48.7 s (8-bit), 2.85× | SELF, setting không rõ | [arXiv 2407.19547](https://arxiv.org/pdf/2407.19547) |
| SDXL trên AGX Orin (MLPerf, TensorRT): ~12.9 s/ảnh | SRC | [NVIDIA Jetson Benchmarks](https://developer.nvidia.com/embedded/jetson-benchmarks) |
| Planner diffusion trên AGX Orin: chạy định kỳ, giảm step/độ phân giải để giữ latency | SRC | [arXiv 2605.05960](https://arxiv.org/pdf/2605.05960) |
| Survey edge: latency là thách thức chính, nhiều op kém hiệu quả trên accelerator biên | SRC | [arXiv 2504.15298](https://arxiv.org/abs/2504.15298) |
| YOLOv8 Orin Nano: TRT FP16 4.43 ms vs PyTorch FP32 20.88 ms (không phải diffusion, chỉ minh hoạ lợi ích TRT) | SELF (cộng đồng) | [benchmarks repo](https://github.com/hokwangchoi/jetson-orin-nano-benchmarks) |

## 6. Áp dụng cho AMR bệnh viện `[INF]`

Toàn bộ mục này là suy luận của tôi từ các claim trên, chưa thử nghiệm.

| Nhu cầu của bạn | Gợi ý | Vì sao |
|---|---|---|
| Né vật cản động | **Giữ NMPC làm lớp an toàn cuối**; diffusion/flow chỉ đề xuất trajectory ứng viên (hoặc warm-start) cho NMPC | Diffusion không có đảm bảo ràng buộc cứng; NMPC cho smooth trajectory và failsafe |
| Navigation học được | Tham khảo NoMaD (goal masking), nhưng đo trên Orin Nano trước khi tin vào "15×" | Số liệu do tác giả tự báo, phần cứng khác |
| Latency | Ưu tiên flow matching 1–10 bước + FP16/INT8 TensorRT; chạy planner ở tần số thấp hơn control loop (STM32/NMPC giữ tần số cao) | Các nguồn §4–§5 cùng chỉ ra step count là đòn bẩy chính |
| Thiếu dữ liệu YOLOv11 | Dùng text-to-image/inpainting (kiểu ROSIE/GenAug) để tăng dữ liệu đối tượng/ánh sáng hành lang; **phải giữ test set toàn ảnh thật** | Tránh đánh giá bị thổi phồng bởi ảnh sinh |
| Failsafe | Timeout inference → rơi về planner cổ điển/dừng an toàn; watchdog trên STM32 | Latency diffusion biến thiên |

## Báo cáo bàn giao

**Trạng thái: PARTIAL**

| AC | Kết quả | Evidence |
|---|---|---|
| AC-01 | PASS (mức tóm tắt) | Mỗi claim có URL; nhưng chỉ dựa trên kết quả search, không đọc toàn văn |
| AC-02 | PASS | Mục 1–3 |
| AC-03 | PASS | Nhãn SELF/INF/`[ ? ]`, hai chỗ mâu thuẫn ghi rõ |
| AC-04 | PASS (suy luận) | Mục 6, chưa thực nghiệm |

- **Kiểm đã chạy:** tự kiểm bằng cách đối chiếu chéo giữa các kết quả search. Không phải kiểm độc lập.
- **Chưa làm:** chưa chạy engine Refinery (`refinery.py`/`bites.py`), vì engine yêu cầu snapshot văn bản + `evidence_span` mà tôi chưa lưu; chưa đọc toàn văn paper.
- **Cần bạn quyết định:** (1) có muốn bước 2: tải snapshot các paper chính và dựng registry Refinery đầy đủ không; (2) trọng tâm tiếp theo: Diffusion Policy cho navigation, hay sinh ảnh tăng dữ liệu cho YOLOv11?
