# Quét độc lập trước khi xem pre-label

Frame: frame_0227.jpg

Số xe nhìn thấy bằng mắt: 25 xe

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Xe tối màu ở làn trong cùng bên trái sát dải phân cách: Do chìm trong vùng bóng tối của dải phân cách và góc khuất ánh sáng đèn xe đối diện, AI rất dễ bỏ sót hoàn toàn (False Negative) hoặc chỉ phát hiện được cụm đèn hậu nhỏ thay vì bao trọn thân xe.
2. Các xe ở cự ly xa gần chân cầu vượt phía trên và xe làn phải có vệt đèn pha chiếu sáng: Cụm xe ở xa có kích thước nhỏ chìm vào ánh đèn đô thị phức tạp, còn các xe có đèn pha rọi sáng mặt đường ướt dễ khiến AI nhận nhầm vệt phản quang là một phần của thân xe dẫn đến box bị vẽ quá to.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
