# Endterm Deep Learning

Repository này lưu trữ toàn bộ mã nguồn, báo cáo và tài liệu trình bày cho đồ án cuối kỳ môn Deep Learning. Đề tài tập trung vào việc ứng dụng các mô hình học sâu hiện đại cho hai nhóm bài toán xử lý thực thể:

- **Task 1 - Visual Question Answering (VQA):** nhận diện, mô tả và suy luận về thực thể động vật trong ảnh thông qua câu hỏi tiếng Việt.
- **Task 2 - Aspect-Based Sentiment Analysis (ABSA):** phân tích sắc thái cảm xúc theo từng khía cạnh trong đánh giá khách sạn tiếng Việt.

Mục tiêu chung của đồ án là khảo sát cách các mô hình nền tảng, kỹ thuật fine-tuning hiệu quả tham số và học tăng cường có thể hỗ trợ các bài toán đa phương thức và ngôn ngữ tự nhiên trong bối cảnh tiếng Việt.

## Tài liệu chính

- `52300115_52300135_52300138.docx`: báo cáo hoàn chỉnh của đồ án, trình bày cơ sở lý thuyết, mô hình đề xuất, quy trình thực nghiệm, phân tích kết quả, hạn chế và hướng phát triển.
- `presentation.pptx`: slide thuyết trình tóm tắt nội dung chính của đề tài.
- `README.md`: file giới thiệu tổng quan repository.

## Cấu trúc thư mục

```text
Endterm_DL/
├── README.md
├── 52300115_52300135_52300138.docx
├── presentation.pptx
└── source_code/
    ├── Task1/
    │   ├── A1/
    │   ├── A2/
    │   ├── B1/
    │   ├── B2/
    │   └── B2_RL/
    ├── Task2/
    └── dataset/
        ├── task1/
        └── task2/
```

## Task 1: Visual Question Answering

Task 1 nghiên cứu bài toán trả lời câu hỏi thị giác trên tập dữ liệu động vật. Với mỗi ảnh, mô hình nhận một câu hỏi tiếng Việt và sinh ra câu trả lời tương ứng. Bài toán này yêu cầu mô hình kết hợp năng lực nhận diện hình ảnh, hiểu ngôn ngữ tự nhiên và sinh câu trả lời phù hợp với ngữ cảnh.

Thư mục `source_code/Task1` gồm nhiều hướng tiếp cận khác nhau, từ các mô hình cơ bản đến mô hình nền tảng đa phương thức:

### A1 - VQA với kiến trúc cơ bản

`source_code/Task1/A1/vqa_a1_model.ipynb`

Notebook này thể hiện hướng tiếp cận VQA truyền thống hơn, trong đó đặc trưng hình ảnh và biểu diễn câu hỏi được kết hợp để sinh hoặc phân loại câu trả lời. Đây là phần nền tảng giúp so sánh với các mô hình lớn hơn ở các cấu hình sau.

### A2 - VQA với kiến trúc cải tiến

`source_code/Task1/A2/vqa_a2_model.ipynb`

Notebook này tiếp tục phát triển hướng A1 bằng kiến trúc mạnh hơn cho việc kết hợp thông tin ảnh và văn bản. A2 đóng vai trò là bước trung gian để đánh giá hiệu quả của các mô hình tự xây dựng trước khi chuyển sang mô hình nền tảng đa phương thức.

### B1 - PaliGemma Zero-shot

`source_code/Task1/B1/B1_PaliGemma_Zeroshot.ipynb`

B1 sử dụng mô hình `PaliGemma-3B` ở trạng thái zero-shot. Mục tiêu là đánh giá năng lực ban đầu của mô hình nền tảng khi chưa được tinh chỉnh trên tập dữ liệu VQA tiếng Việt của đồ án. Phần này cho thấy mô hình có khả năng hiểu hình ảnh và câu hỏi ở mức nhất định, nhưng còn gặp hạn chế trong việc căn chỉnh đầu ra sang tiếng Việt.

### B2 - PaliGemma Fine-tuning

`source_code/Task1/B2/B2_PaliGemma_FineTune.ipynb`

B2 là hướng tiếp cận chính cho bài toán VQA. Notebook này tinh chỉnh PaliGemma bằng kỹ thuật QLoRA, giúp mô hình thích nghi tốt hơn với dữ liệu động vật và câu trả lời tiếng Việt trong khi vẫn tiết kiệm tài nguyên tính toán. Giai đoạn này cải thiện đáng kể khả năng trả lời theo đúng miền dữ liệu so với zero-shot.

### B2_RL - PaliGemma DPO

`source_code/Task1/B2_RL/B2_PaliGemma_DPO.ipynb`

B2_RL thử nghiệm bổ sung Direct Preference Optimization (DPO) sau giai đoạn fine-tuning. Ý tưởng là dùng dữ liệu preference gồm các cặp câu trả lời tốt và kém để căn chỉnh mô hình theo hướng sinh câu trả lời phù hợp hơn. Kết quả thực nghiệm cho thấy hướng DPO triển khai được về mặt kỹ thuật, nhưng trong cấu hình hiện tại chưa tạo ra cải thiện rõ rệt cho mô hình VQA.

## Task 2: Aspect-Based Sentiment Analysis

Task 2 xử lý bài toán phân tích sắc thái cảm xúc theo khía cạnh trên dữ liệu đánh giá khách sạn tiếng Việt. Thay vì chỉ xác định cảm xúc chung của toàn câu, mô hình cần nhận diện cảm xúc ứng với từng khía cạnh như vị trí, giá cả, dịch vụ, phòng ở hoặc tiện nghi.

Thư mục `source_code/Task2` được tổ chức theo các giai đoạn thực nghiệm:

### Stage 1 - PhoBERT SFT

`source_code/Task2/stage1_PhoBERT_ABSA.ipynb`

Giai đoạn đầu sử dụng PhoBERT làm mô hình nền cho phân loại cảm xúc theo khía cạnh. Đây là bước học có giám sát chính, giúp mô hình nắm được quan hệ giữa câu đánh giá, khía cạnh và nhãn cảm xúc tương ứng.

### Stage 2 - Critic Pretraining

`source_code/Task2/stage2_PhoBERT_ABSA.ipynb`

Giai đoạn này chuẩn bị thành phần critic cho hướng học tăng cường. Critic có nhiệm vụ ước lượng chất lượng của hành động hoặc dự đoán, từ đó hỗ trợ quá trình tối ưu ở giai đoạn PPO.

### Stage 3 - PPO Optimization

`source_code/Task2/stage3_PhoBERT_ABSA.ipynb`

Giai đoạn cuối thử nghiệm Proximal Policy Optimization (PPO) để tối ưu mô hình dựa trên tín hiệu phần thưởng. PPO được dùng như một cơ chế căn chỉnh nhằm kiểm soát quá trình cập nhật và hạn chế mô hình lệch quá xa so với mô hình đã học có giám sát.

### Demo

`source_code/Task2/demo_PhoBERT_ABSA.ipynb`

Notebook demo minh họa cách mô hình ABSA xử lý câu đánh giá và đưa ra dự đoán cảm xúc theo khía cạnh. Đây là phần phục vụ trình bày kết quả và kiểm tra trực quan đầu ra của hệ thống.

## Dữ liệu

Thư mục `source_code/dataset` chứa thông tin và dữ liệu liên quan đến hai task:

- `source_code/dataset/task1/readme.txt`: ghi chú về dữ liệu dùng cho bài toán VQA.
- `source_code/dataset/task2/task2_dataset.zip`: dữ liệu dùng cho bài toán ABSA.

Do giới hạn dung lượng và cách tổ chức dữ liệu trong quá trình thực nghiệm, một số notebook có thể tham chiếu đến dữ liệu được lưu trên Google Drive trong môi trường Colab. Báo cáo chính mô tả đầy đủ hơn về nguồn dữ liệu, cách xử lý và vai trò của từng tập dữ liệu trong thí nghiệm.

## Tổng quan phương pháp

Đồ án kết hợp nhiều hướng tiếp cận khác nhau:

- Các mô hình VQA cơ bản để xây dựng baseline ban đầu.
- Mô hình thị giác - ngôn ngữ PaliGemma cho bài toán VQA đa phương thức.
- QLoRA để fine-tune mô hình lớn với chi phí tài nguyên thấp hơn.
- DPO để khảo sát khả năng căn chỉnh mô hình VQA bằng dữ liệu preference.
- PhoBERT cho bài toán phân tích cảm xúc tiếng Việt theo khía cạnh.
- PPO để thử nghiệm hướng học tăng cường trong bài toán ABSA.

Thông qua các thực nghiệm này, repository phản ánh quá trình phát triển từ mô hình cơ bản đến mô hình nền tảng, từ học có giám sát đến các kỹ thuật căn chỉnh nâng cao. Các kết quả cho thấy fine-tuning vẫn là thành phần quan trọng nhất để thích nghi mô hình với dữ liệu tiếng Việt, trong khi các phương pháp học tăng cường như DPO và PPO cần dữ liệu phản hồi, hàm thưởng và thiết kế thực nghiệm mạnh hơn để tạo ra cải thiện rõ rệt.

## Kết luận chung

Repository này là sản phẩm tổng hợp của quá trình nghiên cứu và thực nghiệm cho hai bài toán đại diện của Deep Learning hiện đại: hiểu ảnh kết hợp ngôn ngữ và hiểu sắc thái cảm xúc theo ngữ cảnh. Phần VQA cho thấy tiềm năng của các mô hình đa phương thức lớn khi được tinh chỉnh đúng miền dữ liệu. Phần ABSA cho thấy vai trò của mô hình ngôn ngữ tiếng Việt trong việc xử lý dữ liệu văn bản có cấu trúc cảm xúc phức tạp.

Các hạn chế còn lại chủ yếu nằm ở chất lượng dữ liệu, sự đa dạng của cách diễn đạt tiếng Việt, mất cân bằng nhãn và hiệu quả chưa ổn định của các bước căn chỉnh bằng học tăng cường. Đây cũng là những hướng có thể tiếp tục phát triển trong các nghiên cứu sau.
