# RareSimDiff MVP: task report (AI Work Method)

Ngày: 2026-11-21 · Chủ việc: người dùng · Phạm vi: chuẩn bị đầy đủ trên máy local (CPU), **dừng trước lần train GPU chính thức**.

## TIP

```text
ID / Chủ việc:        RARESIMDIFF-MVP / người dùng
Mục tiêu:             Một diffusion model có điều kiện (RareScenarioDiT) học phân phối scene mô phỏng + tham số rare-event cho humanoid
                      robot learning; sinh scenario thực thi được trong MuJoCo; train đúng MỘT lần trên RTX 4090.
Đầu ra / Nơi lưu:     raresimdiff/ (mã, test, docs), dist/raresimdiff_gpu_bundle.tar.zst + .sha256, báo cáo này.
Input / Scope:        3 tài liệu thiết kế 2026-11-05 trong docs/ai-native/tasks/; prompt đặc tả; không có humanoid model trong repo.
Giới hạn:             Không train GPU; không push GitHub; không chạy nhiều kiến trúc; không bịa số đo GPU.
Được tự làm:          Chọn kích thước dataset, robot proxy, ngưỡng validation, cấu hình huấn luyện (ghi lại để duyệt).
Cần chủ việc quyết:   Duyệt dataset + config + ngân sách (docs/RUN_APPROVAL.md); cung cấp humanoid MJCF/URDF thật nếu muốn validate humanoid.
Giả định:             GPU máy ngoài có CUDA đủ mới để chạy wheel torch; nếu không, scripts/setup_env.sh dò và thử wheel khác.
```

Tiêu chí nghiệm thu (AC) và kết quả nằm ở phần "Báo cáo bàn giao" cuối file.

## Tiêu chí nghiệm thu (AC)

| AC | Tiêu chí quan sát được | Cách kiểm |
|---|---|---|
| AC-01 | Có triển khai chạy được (không chỉ tài liệu): scenario generator, MuJoCo backend, diffusion model, trainer, generate CLI, policy integration | 49 test + chạy end-to-end CPU |
| AC-02 | Gate G0–G5 đều GO trên máy local | `scripts/local_preflight.sh` → `dist/gates_local.json` |
| AC-03 | Dataset ≥ 100k scenario hợp lệ, có checksum, split theo parent, thống kê chỉ từ train | `verify_dataset` (13 kiểm) |
| AC-04 | Bundle GPU + sha256 + gate G6 GO; bundle giải nén chạy được kiểm tra toàn vẹn và dừng đúng khi không có GPU | `scripts/package_for_gpu.sh`, giải nén thử, thử làm hỏng file |
| AC-05 | KHÔNG có bước tối ưu GPU nào được chạy; không có số đo GPU bịa | `preflight` ghi `optimizer_steps_taken: 0`; báo cáo ghi rõ ESTIMATED |
| AC-06 | Resume tái lập chính xác; run hoàn tất không bị ghi đè | test bitwise + test run-dir |
| AC-07 | Tách phần đã triển khai khỏi phần "paper-inspired" | `docs/ARCHITECTURE.md` bảng Implemented vs paper-inspired |

## Quyết định đã đưa ra (cần chủ việc xác nhận khi duyệt)

| Quyết định | Lý do | Tác động |
|---|---|---|
| Robot **proxy** (mobile base + reach + lift + gripper), không phải humanoid | Repo không có MJCF/URDF humanoid (G0 xác nhận) | Validate humanoid = **BLOCKED**; `SimulatorBackend` là điểm cắm cho humanoid/Isaac |
| Layout rời rạc (asset, mask, env) là *conditioning*, bootstrap từ train | Yêu cầu: categorical không diffuse như liên tục | Scene sinh ra có tham số liên tục mới nhưng layout lặp lại từ train |
| Perception degradation chỉ đổi observation | Phân biệt cosmetic vs dynamics; không có renderer | Không có hiệu ứng pixel |
| Nhãn dataset đến từ baseline scripted (không phải expert) | Không có expert/teleop; cấm bịa action | Không xuất demo expert; `is_expert_demo=False` |
| Dataset 100k scenario (25.000 parent × 4 variant), 30.000 bước train | Đo throughput pilot 70/s; ~420 epoch trên 73k mẫu | Có rủi ro overfit → chọn best theo val-EMA + đo memorization trong evaluation |
| Bundle không kèm trajectories (13 MB, còn ở `data/.../trajectories`) | Train không cần; giảm kích thước | Dùng `--include-trajectories` nếu cần |
| Không push GitHub | Yêu cầu của prompt | Chỉ commit local |

## Báo cáo bàn giao

**Trạng thái: PARTIAL** (phần CPU hoàn tất và kiểm chứng; chưa có model đã huấn luyện, chưa kiểm CUDA path, chưa validate humanoid thật).
**Trạng thái GO/NO-GO của dự án: `READY_FOR_GPU_TRAINING`** (chờ người dùng duyệt `docs/RUN_APPROVAL.md`; train chính thức chưa được chạy).

| AC | Kết quả | Evidence |
|---|---|---|
| AC-01 | PASS | 49/49 test pass; pipeline end-to-end CPU (procedural → encode → diffusion → decode → validate → episode) |
| AC-02 | PASS | G0–G5 GO (`dist/gates_local.json`): tests G1 14, G2 12, G3 3, G4 14, G5 9, đều 0 fail |
| AC-03 | PASS | 99.717 scenario hợp lệ (train 73.295 / val 9.325 / test 9.016 / ood_test 8.081); 13/13 kiểm integrity |
| AC-04 | PASS | G6 GO 11/11; sha256 `11292c11acd9dab904e488db2f2318289d33a6b5c93d43fef674e0179088adef`; giải nén thử: integrity OK, preflight dừng exit 2 vì không có CUDA; sửa 1 file → exit 1 |
| AC-05 | PASS | Không chạy GPU; preflight local ghi 0 bước tối ưu; mọi số GPU ghi ESTIMATED |
| AC-06 | PASS | `test_resume_is_exact_continuation` (tham số + EMA giống hệt bit-by-bit); `test_run_dir_immutability_and_resume_rules`; chạy lần 2 trên run hoàn tất → REFUSED exit 3 |
| AC-07 | PASS | `docs/ARCHITECTURE.md` |
| **GPU/CUDA code path** | **NOT_CHECKED** | bf16 autocast, generator CUDA, `scripts/setup_env.sh` (chọn wheel theo driver), cờ deterministic trên CUDA chưa chạy được ở đây (không có GPU). Giảm rủi ro: preflight chạy fwd/bwd trên GPU không bước tối ưu trước khi train |
| **Validate humanoid** | **BLOCKED** | Không có humanoid model trong repo |

### Kiểm đã chạy (tự kiểm, không phải kiểm độc lập)
* `pytest`: 49 passed, 0 failed (unit schema/codec 11, diffusion 7, physics 12, training 7, integration 9, regression 3). Gate G0–G5 chạy lại sau chỉnh sửa mã cuối; sau đó chỉ sửa tài liệu và chạy lại toàn bộ `pytest` lần cuối (49 passed) và gate G6.
* Physics: reset/step tất định (bit-by-bit); vật nằm trên bàn sau settle, penetration ≤ 4 mm; event friction/force đúng thời điểm; obstacle chuyển động liên tục, không teleport; perception model đúng công thức, không đổi vật lý; không NaN trên 75 scenario; baseline: nominal thành công ≥ 9/10, slip nặng thất bại nhiều hơn slip nhẹ.
* Gradient hữu hạn, overfit batch nhỏ trên CPU giảm loss > 65 %, masking, conditioning, DDIM tất định theo seed.
* CPU smoke (cấu hình model chính thức, 300 bước, 7.524.382 tham số): loss huấn luyện 0,527 (bước 1–100) → 0,321 (bước 201–300), val-EMA 0,3176, 0 bước bị bỏ qua, ~1.000 mẫu/s trên 4 CPU. Evaluation chạy hết (n=40/điều kiện): valid ratio learned 0,80 vs procedural 0,95; event accuracy 0,998. **Đây chỉ là smoke test, không phải bằng chứng chất lượng.**

### Kết quả dataset (đo thật)
* Sinh 100.000 → giữ 99.717; loại 283: OBSTACLE_OVERLAP 235, UNSTABLE_SETTLE 40, EPISODE_UNSTABLE 7, DUPLICATE 1. 71,4 scenario/s (4 CPU worker, 1.400 s); pilot 1.200 scenario đo trước: 70/s.
* Tỷ lệ thành công của baseline theo severity (train): object_slip 0,93 / 0,77 / 0,40 / 0,12; moving_obstacle 1,00 / 0,95 / 0,67 / 0,25; perception 0,94 / 0,67 / 0,36 / 0,26; slip+perception 0,90 / 0,54 / 0,19 / 0,05 (4 bin 0–0,25 … 0,75–1). `ood_test` thành công 0,12 (khó hơn rất nhiều so với 0,60 của train).
* Hạn chế: moving_obstacle bin severity thấp nhất chỉ 550 mẫu (severity thấp rất hiếm với obstacle vì ép obstacle băng qua hết hành lang) → generator có thể không đạt severity < 0,25 cho obstacle; `support_warnings` sẽ cảnh báo OOD khi điều kiện nằm ngoài khoảng train.

### Chưa kiểm / mâu thuẫn / rủi ro còn lại
1. Chất lượng của model sau train chính thức: chưa biết. 420 epoch trên 73k mẫu có thể memorization; evaluation đo `memorization_rate` và novelty.
2. Giá trị "learned thêm coverage so với procedural": chưa xác định; cần chạy `evaluation_report` sau train và nghiên cứu downstream (EXPERIMENT_PROTOCOL §B, chưa chạy).
3. Không có policy/expert nào; không có kết quả policy nào được báo.
4. Không tuyên bố sim-to-real.
5. Thời gian train trên 4090: **không biết**; bộ nhớ ước tính 1–3 GB (ESTIMATED), preflight sẽ đo thật.

### Cần chủ việc
1. Duyệt `raresimdiff/docs/RUN_APPROVAL.md` (dataset fingerprint `b570009f…`, training hash `2236948f…`, 30.000 bước, 1 run). Bundle cuối: sha256 `51310dfd0b5c4c2025faa21bb345278f3bdcc2a968a2e5a1d86ec096fd55ef0e` (xem phần Release audit; con số `11292c11…` ở trên là bản trước audit, đã bị thay thế).
2. Cung cấp humanoid MJCF/URDF nếu muốn validate humanoid thật.
3. Xác nhận có commit/push bundle hoặc nhánh lên GitHub không (hiện chỉ commit local; `dist/`, `data/` bị `.gitignore`).

## Release audit (2026-11-21, trước khi push)

Kiểm tra trên **bundle sẽ push** (`raresimdiff_gpu_bundle.tar.zst`, sha256 `51310dfd0b5c4c2025faa21bb345278f3bdcc2a968a2e5a1d86ec096fd55ef0e`, 101 file, 8.913.840 byte):

| Kiểm | Kết quả | Evidence |
|---|---|---|
| Nội dung đủ: train/val/test/ood_test, preprocessing stats, 4 manifest split, configs, source (≥ 50 file .py), scripts | PASS | `required_content_present` |
| Hash dataset trong bundle = `dataset_manifest.json` = thư mục dataset gốc; fingerprint `b570009f46e8…` | PASS | so từng file; `verify_dataset` trên bản giải nén: 13/13 |
| Normalization stats tính từ train (`computed_from: train`, hash khớp) | PASS | `preprocessing_stats.json` |
| Dataset không bị đổi trong lúc audit | PASS | sha256 trước/sau trùng nhau |
| Secret/credential/đường dẫn nhạy cảm | PASS (có allowlist đã xem xét) | quét 101 file + từng member trong `.npz`; 0 hit sau allowlist: 2 mẫu regex trong `gates.py` (chính bộ quét) và URL repo GitHub công khai trong `GPU_DEPLOY.md` (không phải credential) |
| Dữ liệu không được phép công khai | PASS | 100 % dataset là scenario mô phỏng sinh thủ tục; trường chuỗi trong `.npz` chỉ là id `p…`; không có dữ liệu thật/cá nhân |
| Epoch tính lại | 419,1 theo mẫu; 419,6 theo cách trainer đếm (drop-last) | global batch hiệu dụng 512 × 2 = 1024; 30.000 × 1024 = 30.720.000 mẫu / 73.295; trainer: 143 micro-batch/epoch, 60.000 micro-batch |

Ghi chú: báo cáo trước ghi "≈ 420 epoch"; số chính xác là 419,1/419,6. Không đổi cấu hình model, không tạo lại dataset, không chạy GPU.
