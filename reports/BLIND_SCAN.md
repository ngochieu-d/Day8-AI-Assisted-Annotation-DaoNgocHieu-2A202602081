# Quét độc lập trước khi xem pre-label

Frame: frame_0182.jpg

Số xe nhìn thấy bằng mắt: 25 xe (kèm 1 vật thể nghi ngờ là xe thứ 26 ở vùng bóng tối)

Hai vị trí dễ bị AI bỏ sót hoặc vẽ sai, kèm mô tả xe:
1. Xe màu tối ngay dưới cùng chính giữa khung hình (foreground sát đáy ảnh): Xe chỉ thấy phần nóc và kính sau lờ mờ chìm trong bóng tối, không thấy rõ đèn pha nên AI rất dễ bỏ sót hoàn toàn hoặc không nhận diện được viền thân xe.
2. Các xe ở xa trên làn đường sát chân cầu vượt phát sáng và xe bị cắt mép ảnh: Cụm xe ở xa kích thước nhỏ chìm trong ánh sáng phức tạp của nền đô thị và chân cầu; các xe ở mép ảnh di chuyển nhanh và bị cắt ngang thân xe dễ bị vẽ lệch hoặc nhận nhầm với vệt đèn pha rọi trên mặt đường.

Chạy `python3 tools/lock_blind.py` ngay sau khi điền. Sau đó giữ file này nguyên vẹn.
