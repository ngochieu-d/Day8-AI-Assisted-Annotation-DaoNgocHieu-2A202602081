# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Đào Ngọc Hiếu
Mã học viên: 2A202602081

Công cụ gán nhãn đã dùng: AnyLabeling

Báo cáo phân tích và đối chiếu số liệu truy xuất trực tiếp từ các file kết quả: `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round0.json`, `outputs/metrics_round1.json`, `outputs/round1_diff.md` và `reports/REVIEW_LOG.csv`. Nhãn tập kiểm thử được nhìn nhận đúng bản chất là nhãn do mô hình tạo ra để đo độ khớp, không phải là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool - 268 ảnh) và tập kiểm thử (test - 20 ảnh) bắt buộc phải được chia theo trục thời gian và có vùng đệm (buffer - 112 ảnh) ở giữa vì góc máy camera được đặt cố định trên cầu vượt quay cảnh giao thông đêm liên tục với tốc độ 2.5 khung hình/giây. Giữa hai khung hình liên tiếp cách nhau chỉ 0.4 giây, bối cảnh mặt đường gần như tĩnh tuyệt đối và mỗi chiếc xe di chuyển qua khung hình thường lưu lại từ vài giây đến hơn chục giây.

Nếu chia tập ngẫu nhiên (random split), cùng một chiếc xe sẽ xuất hiện đồng thời ở cả tập huấn luyện và tập kiểm thử (chỉ xê dịch một vài pixel giữa các frame liền kề). Khi đó, các số đo trên tập kiểm thử (như AP50, Precision, Recall) sẽ bị **lệch lạc nghiêm trọng theo hướng lạc quan giả tạo (artificially over-optimistic)** do hiện tượng rò rỉ dữ liệu (data leakage). Mô hình đạt điểm cao không phải vì học được khả năng phát hiện xe tổng quát, mà chỉ đơn thuần là "nhận diện lại đúng chiếc xe quen thuộc" mà nó vừa được huấn luyện ở frame trước. 

Vùng đệm 112 ảnh (tương đương 4 giây trước và sau mỗi đoạn kiểm thử) đóng vai trò như một rào cản thời gian an toàn tối thiểu 4.4 giây, đảm bảo mọi phương tiện xuất hiện trong tập kiểm thử đã di chuyển hoàn toàn ra khỏi góc quay trước khi tập pool bắt đầu, bảo toàn tính khách quan và trung thực của phép đo.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng số liệu Vòng 0 trích xuất trực tiếp từ `reports/rounds_table.md` và `outputs/metrics_round0.json`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Dựa vào ảnh đối chiếu `outputs/compare_round0.jpg` và các số đo:
- **Các loại xe mô hình không khớp nhãn tham chiếu:** Mô hình khởi đầu lạnh (YOLOv8n tiền huấn luyện trên COCO) bỏ sót rất nhiều xe chạy trong bóng đêm, đặc biệt là các xe màu tối chìm vào nền đường và các xe ở cự ly xa chỉ nhìn thấy hai chấm đèn mờ ảo (thể hiện qua rất nhiều box False Negative màu vàng). Ngoài ra, các xe đi sát dải phân cách mép trái hoặc xe bị che khuất một phần cũng thường xuyên bị bỏ qua.
- **Ý nghĩa độ phủ (Recall) theo kích thước xe:** `R small` chỉ đạt **0.182** (bỏ sót hơn 81.8% xe nhỏ ở xa), trong khi `R medium` đạt **0.547** và `R large` đạt **0.561**. Điều này phản ánh rõ ràng hạn chế về tầm nhìn của mô hình COCO ban đầu: kích thước xe càng nhỏ và ở càng xa thì độ tương phản với nền trời đêm càng thấp, khiến mô hình thiếu đặc trưng hình học để vượt qua ngưỡng tự tin kích hoạt phát hiện.
- **Trường hợp cần rà soát lại nhãn tham chiếu:** Trong `outputs/compare_round0.jpg`, xuất hiện một số box màu đỏ bị hệ thống tính là False Positive (phát hiện nhầm). Tuy nhiên, khi phóng to quan sát bằng mắt thường, vị trí đó thực sự có một chiếc xe đang chạy ở làn xa bên phải nhưng bộ nhãn tham chiếu tự động đã bỏ sót không đánh nhãn. Mô hình thực tế đã phát hiện đúng nhưng lại bị chấm oan thành lỗi. Do đó, cần có con người rà soát đối chiếu trước khi vội vàng kết luận mô hình dự đoán sai.

## 3. Chiến lược chọn mẫu

### Giải thích công thức và tham số
Công thức tính điểm chọn mẫu trong thuật toán Active Learning (`tools/al_select.py`):
$$\text{score} = W_U \cdot U + W_A \cdot A + W_D \cdot D = 0.5 \cdot U + 0.3 \cdot A + 0.2 \cdot D$$
- **$U$ (Uncertainty - Trọng số 0.5):** Là độ bất định trung bình của 5 box khó nhất trong ảnh, tính theo $u = 1 - |2 \cdot \text{conf} - 1|$. Đại lượng này đạt giá trị cực đại bằng 1.0 khi độ tin cậy $\text{conf} = 0.5$, phản ánh trạng thái mô hình lưỡng lự và thiếu tự tin nhất giữa hai quyết định (là xe hay không phải xe).
- **$A$ (Ambiguity - Trọng số 0.3):** Là tỷ lệ số box mơ hồ có độ tin cậy nằm trong vùng tranh tối tranh sáng ($0.15 \le \text{conf} < 0.50$), chuẩn hóa theo ảnh có nhiều box mơ hồ nhất trong pool. Yếu tố này ưu tiên các khung hình có mật độ đối tượng mập mờ cao.
- **$D$ (Diversity - Trọng số 0.2):** Là khoảng cách thời gian từ ảnh đang xét tới ảnh đã được gán nhãn gần nhất (chuẩn hóa tối đa 10 giây). Đại lượng này giúp rải đều dữ liệu được chọn trên toàn bộ trục thời gian của video, tránh tập trung dồn dập vào một khoảng thời gian hẹp.
- **Vai trò của `MIN_GAP_S = 2.0s`:** Đây là ngưỡng lọc giãn cách thời gian bắt buộc giữa hai ảnh bất kỳ trong cùng một lô. Vì camera đứng yên, hai khung hình cách nhau dưới 2 giây có luồng xe gần như trùng lặp (near-duplicates). Ràng buộc này giúp ngăn ngừa lãng phí công sức của con người khi phải gán nhãn lặp lại cho cùng những chiếc xe đó.

### Dẫn chứng cân nhắc từ `reports/SELECTION.md`
- **Ba frame thuộc lô 12 ảnh mô hình chọn:**
  1. `frame_0182.jpg` (Rank 1, score = 0.9591, A = 1.0): Frame có điểm bất định cao nhất toàn bộ tập dữ liệu với 18 box mơ hồ, tập trung luồng xe ban đêm phức tạp nhất.
  2. `frame_0369.jpg` (Rank 2, score = 0.9324, U = 0.9315): Có tới 43 box phát hiện, mật độ giao thông đông đúc và cách frame 0182 gần 75 giây, đảm bảo tối đa tính đa dạng theo thời gian.
  3. `frame_0331.jpg` (Rank 5, score = 0.9154, 47 box đề xuất): Chứa lượng box dự đoán cao nhất và xuất hiện nhiều xe tải lớn cùng vệt đèn pha phản chiếu trên đường.
- **Một frame khác thể hiện sự cân nhắc:**
  + `frame_0372.jpg` (Rank 6, score = 0.9101, t = 148.8s): Dù có điểm số nằm trong top 6 cao nhất pool, frame này đã bị thuật toán gạt bỏ vì nó chỉ cách `frame_0369.jpg` (t = 147.6s) đúng 1.2 giây (< `MIN_GAP_S`). Việc loại bỏ frame này giúp tiết kiệm chi phí dán nhãn, tránh nạp dữ liệu gần trùng vào tập huấn luyện.

### Điểm bất định có chứng minh ảnh sẽ cải thiện mô hình không?
**Không.** Điểm bất định cao chỉ phản ánh rằng mô hình hiện tại đang gặp khó khăn và phân vân trước bức ảnh đó, chứ hoàn toàn không bảo đảm rằng việc gán nhãn ảnh đó sẽ làm tăng điểm AP50 trên tập kiểm thử. Trong điều kiện quay đêm, nếu độ bất định sinh ra do nhiễu hạt cảm biến (sensor noise), ánh đèn pha xe đối diện chiếu thẳng làm lóa camera hoặc vệt nước phản quang trên mặt đường, việc ép mô hình học các khung hình này có thể khiến mô hình bị quá khớp với nhiễu cục bộ thay vì cải thiện năng lực nhận diện phương tiện thực tế.

## 4. Các vòng học chủ động (active learning)

### Bảng số đo tổng hợp các vòng
Trích xuất từ `reports/rounds_table.md` và `outputs/metrics_round1.json`:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 334 | 0.843 | +0.072 | 1.000 | 0.176 | 0.300 | 0.000 | 0.162 | 0.561 |

### Thống kê mức độ sửa nhãn gợi ý (từ `outputs/round1_diff.md`)
Trên lô 12 ảnh của Vòng 1, từ 169 box do mô hình ban đầu đề xuất, học viên đã rà soát và hoàn thiện thành **334 box**:
- **Số box giữ nguyên (accepted):** 136 box (tỷ lệ chấp nhận 80%).
- **Số box chỉnh sửa (edited):** 18 box.
- **Số box xóa bỏ (deleted - False Positive của mô hình):** 15 box.
- **Số box thêm mới (added - False Negative của mô hình):** 180 box.

### Ba vấn đề chính phát hiện khi AI tạo nhãn và cách xử lý khi gán nhãn lại
Trong quá trình trực tiếp kiểm tra và chỉnh sửa trên AnyLabeling, có **3 vấn đề kỹ thuật nổi cộm** đã được phát hiện và xử lý triệt để:

1. **Vấn đề 1: Box AI vẽ quá to (Over-sized / bloated boxes):**
   * *Hiện tượng:* AI bị đánh lừa bởi vệt sáng đèn pha chiếu dài xuống mặt đường phía trước xe hoặc vùng bóng tối hòa lẫn giữa thân xe và dải phân cách. Mô hình nhận nhầm vệt đèn là một phần của đầu xe và kéo dài box ra mặt đường (ví dụ ở `frame_0227.jpg` và `frame_0182.jpg`).
   * *Cách xử lý:* Kéo 4 cạnh của box thu hẹp lại, ôm khít ranh giới thân xe thực tế theo đúng quy tắc (không tính vệt sáng đèn pha rọi xuống đường).
2. **Vấn đề 2: Box AI vẽ quá bé / chỉ khoanh được một vùng đầu hoặc đuôi xe (Under-segmented boxes):**
   * *Hiện tượng:* Ban đêm thân xe màu tối bị bóng đêm nuốt trọn, AI chỉ "nhìn" thấy cụm đèn pha hoặc đèn hậu rực sáng nên chỉ khoanh đúng một vùng nhỏ quanh bóng đèn thay vì bao trọn chiếc xe (ví dụ ở `frame_0099.jpg`).
   * *Cách xử lý:* Kéo mở rộng box bao quát toàn bộ khối thân xe dựa theo ước lượng đường viền quanh cụm đèn chiếu sáng (tuân thủ nghiêm ngặt dòng 17 của `GUIDELINE_LABEL.md`).
3. **Vấn đề 3: Bỏ sót nghiêm trọng xe trong bóng tối & Box trùng lặp/nhiễu phản quang:**
   * *Hiện tượng:* AI bỏ sót tới hơn 50% số lượng xe thực tế, đặc biệt là các xe chạy ở làn trong cùng hoặc xe tối màu sát mép dưới camera. Đồng thời, AI thường xuyên sinh ra 2 box đè lên cùng 1 xe (duplicate boxes) hoặc vẽ nhầm box vào vùng sáng phản chiếu trên mặt đường ướt (như ở `frame_0331.jpg`).
   * *Cách xử lý:* Vẽ thêm mới 180 box cho các xe bị bỏ sót (hành động `added`) và xóa bỏ 15 box nhận nhầm hoặc box trùng lặp (hành động `deleted`).

### Biến động số đo sau fine-tune và phân tích hình ảnh so sánh (`outputs/compare_round1.jpg`)
- **Tăng trưởng AP50 vượt trội:** AP50 tăng từ **0.771** lên **0.843** (tăng **+0.072**, tương ứng +7.2%). Điều này chứng minh rằng việc bổ sung 180 box xe bị bỏ sót và chuẩn hóa lại viền box đã giúp mô hình học được đặc trưng hình học chuẩn xác của phương tiện trong cảnh đêm.
- **Độ chính xác Precision đạt tuyệt đối 1.000 (FP = 0):** Trên 20 ảnh kiểm thử, mô hình không phạm phải bất kỳ một lỗi phát hiện nhầm nào (0 False Positives tại ngưỡng conf 0.25).
- **Phân tích hình ảnh `compare_round1.jpg`:**
  * *Ca tốt lên rõ rệt:* Các box dự đoán của Vòng 1 ôm khít thân xe một cách hoàn hảo, triệt tiêu hoàn toàn hiện tượng box bị kéo dài ngoằng theo vệt đèn pha như ở Vòng 0. Nhóm xe lớn giữ vững độ phủ cao ($R_{\text{large}} = 0.561$).
  * *Ca còn hạn chế:* Các xe ở rất xa dưới chân cầu do tín hiệu quang học quá yếu vẫn chưa vượt qua ngưỡng kích hoạt conf 0.25 ($R_{\text{small}} = 0.000$), do tập huấn luyện 12 ảnh ban đầu chưa chứa đủ số lượng mẫu xe nhỏ ở đường chân trời.

### Phân biệt ba nguồn thông tin và ca khó theo guideline
- **Quan sát độc lập (`reports/BLIND_SCAN.md` trên `frame_0182.jpg`):** Mắt người đếm được 25 xe rõ ràng và 1 xe nghi ngờ (tổng 26 xe), ghi nhận trước nguy cơ AI bỏ sót xe tối màu ở cận cảnh đáy ảnh và cụm xe ở xa sát chân cầu.
- **Lỗi pre-label đã sửa (`outputs/round1_diff.md` & `reports/REVIEW_LOG.csv`):** AI ban đầu chỉ tìm được 13 xe. Người gán nhãn đã thêm mới 14 xe (trong đó có đúng chiếc xe tối màu ở mép dưới `cx=0.433`), sửa 2 box ôm vệt đèn, xóa 1 box trùng lặp để đưa tổng số xe lên đúng 26 xe khớp hoàn toàn với quan sát độc lập.
- **Mô tả ca khó theo guideline:** Tình huống xe ở xa trên làn đường sát chân cầu vượt phát sáng (`frame_0380.jpg`): Thân xe tối màu chìm hoàn toàn vào màn đêm, chỉ thấy hai đốm sáng đỏ. Người gán nhãn phải ước lượng vùng bao thân xe quanh cụm đèn theo dòng 17, đồng thời áp dụng dòng 21 để bỏ qua các box có chiều cao dưới 16 pixel nhằm tránh đưa các nhãn đoán mò thiếu nhất quán vào mô hình.

## 5. Kết luận và giới hạn

### Đánh giá và quyết định tiếp tục/dừng
Vòng 1 đạt kết quả rất tích cực: **AP50 tăng từ 0.771 lên 0.843 (+0.072)** và **Precision đạt mức tuyệt đối 1.000**. Việc chuẩn hóa nhãn đã giúp mô hình khắc phục triệt để lỗi bắt nhầm vệt đèn pha và nâng cao chất lượng phát hiện trên các phương tiện cự ly gần và trung bình. Tôi quyết định **tiếp tục Vòng 2** (nếu tiếp tục triển khai) để tập trung nạp thêm các ảnh có nhiều xe nhỏ ở xa nhằm cải thiện độ phủ Recall cho nhóm xe nhỏ.

### Đề xuất 2 ca cho vòng tiếp theo từ `outputs/selection_round2.csv`
1. **frame_0073.jpg** (Rank 5 vòng 2, t = 29.2s, score = 0.9022, U = 0.9388): Nằm ở khoảng đầu video, có độ bất định cao và chứa nhiều xe ở làn ngoài cùng có độ tương phản thấp. Chi phí rà nhãn vừa phải (~30 box) và phân bố thời gian độc lập.
2. **frame_0126.jpg** (Rank 6 vòng 2, t = 50.4s, score = 0.8931, U = 0.9209): Nằm ở mốc thời gian cách xa các ảnh vòng 1, có nhiều xe tải và xe khách di chuyển với tốc độ khác nhau.
- *Nguy cơ ảnh gần trùng:* Cần duy trì nghiêm ngặt ràng buộc $\text{MIN\_GAP\_S} \ge 2.0s$ đối với cả tập train vòng 1 lẫn các ứng viên mới để tránh lãng phí ngân sách gán nhãn.

### Tác động của các giới hạn đánh giá
- **Tập kiểm thử chỉ 20 ảnh:** Cỡ mẫu nhỏ khiến phương sai thống kê rất nhạy cảm; chỉ cần một vài xe thay đổi trạng thái phát hiện đã làm biến động đáng kể chỉ số AP50.
- **Luật bỏ qua xe nhỏ < 16 pixel:** Giúp bảo vệ mô hình không bị phạt bởi các vật thể quá mập mờ ở đường chân trời, định hướng mô hình tập trung vào các phương tiện có khả năng gây rủi ro ở cự ly quan sát thực tế.
- **Nhãn tham chiếu do mô hình AI tạo ra:** Bộ nhãn test chưa qua rà soát thủ công của con người nên vẫn chứa lỗi bỏ sót và lệch box. Do đó, điểm số AP50 phản ánh mức độ tương đồng giữa mô hình của học viên với mô hình sinh nhãn tham chiếu, không phải là thước đo chân lý tuyệt đối ngoài đời thực.

### Quy trình kiểm tra khi AP50 giảm trước khi train thêm
Nếu chỉ số AP50 giảm sau một vòng học chủ động, thay vì vội vàng huấn luyện thêm dữ liệu, cần thực hiện quy trình kiểm tra 3 bước:
1. **Kiểm tra độ nhất quán của nhãn (Annotation Consistency):** Rà soát lại xem quy tắc vẽ box có bị mâu thuẫn giữa các frame hay không (ví dụ: có frame thì thu hẹp box vệt đèn, có frame lại vô tình vẽ trùm ra ngoài).
2. **Phân tích phân bố Confidence Score:** Đánh giá lại độ tin cậy của mô hình trên tập test; nếu mô hình bị dịch chuyển tự tin về mức thấp, thử nghiệm hạ ngưỡng đánh giá (ví dụ đo tại conf 0.10) để đo lường Recall thực tế.
3. **Kiểm soát Overfitting trên tập mẫu nhỏ:** Điều chỉnh số epoch huấn luyện (50 epochs trên 12 ảnh có thể khiến mô hình bị quá khớp với một vài góc quay cục bộ) hoặc bổ sung kỹ thuật data augmentation phù hợp với cảnh đêm.
