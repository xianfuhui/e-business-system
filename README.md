HƯỚNG DẪN CHẠY DỰ ÁN: PROCESS MINING SYSTEM (Spring Boot + Python)
====================================================================
Repo tham khảo: https://github.com/xianfuhui/e-business-system

Cấu trúc thư mục giả định:
  ├── 100 UI/              -> Source code Spring Boot (Java)
  ├── Python/
  │     ├── install.txt     -> Danh sách thư viện Python cần cài
  │     └── run.py          -> File chạy Flask API (xử lý dữ liệu / ML)
  └── Dataset/
        └── dataset.csv     -> File dữ liệu event log đầu vào

--------------------------------------------------------------------
BƯỚC 0: YÊU CẦU HỆ THỐNG
--------------------------------------------------------------------
- Java JDK 17+ (hoặc phiên bản Spring Boot project yêu cầu)
- Maven (hoặc dùng ./mvnw có sẵn trong project)
- Python 3.10 hoặc 3.11 (khuyến nghị, để tương thích TensorFlow)
- Graphviz đã cài ở HỆ ĐIỀU HÀNH (không chỉ pip) — cần cho vẽ sơ đồ
  quy trình (nx_pydot / pm4py). Tải tại: https://graphviz.org/download/
  Sau khi cài, nhớ thêm Graphviz vào biến môi trường PATH.
- (Tuỳ chọn) Khoảng 8GB RAM trống trở lên nếu file CSV lớn.

--------------------------------------------------------------------
BƯỚC 1: CHUẨN BỊ MÔI TRƯỜNG PYTHON
--------------------------------------------------------------------
1. Mở terminal, di chuyển vào thư mục Python:
       cd Python

2. Tạo môi trường ảo (khuyến nghị để tránh xung đột thư viện):
       python -m venv venv

3. Kích hoạt môi trường ảo:
   - Windows (cmd):        venv\Scripts\activate
   - Windows (PowerShell): venv\Scripts\Activate.ps1
   - macOS / Linux:        source venv/bin/activate

4. Cài các thư viện cần thiết từ file install.txt:
       pip install -r install.txt

   Nếu máy không có GPU / gặp lỗi cài tensorflow, có thể thử:
       pip install tensorflow-cpu

--------------------------------------------------------------------
BƯỚC 2: THIẾT LẬP DATASET
--------------------------------------------------------------------
1. Đặt file dữ liệu event log vào:  Dataset/dataset.csv
2. Mở file run.py, kiểm tra/sửa lại đường dẫn biến DATASET_DIR (hoặc
   truyền tên file dataset.csv khi gọi API /process) cho khớp với vị
   trí thật của thư mục Dataset trên máy bạn.
3. File CSV cần có tối thiểu các cột (có thể đổi tên trong code nếu
   dataset khác):
       event_time, event_type, product_id, category_code,
       category_id, brand, price, user_id, user_session

--------------------------------------------------------------------
BƯỚC 3: CHẠY PYTHON API (Flask)
--------------------------------------------------------------------
1. Vẫn trong thư mục Python (đã activate venv):
       python run.py

2. Mặc định server chạy tại: http://localhost:5000
   Kiểm tra bằng cách mở trình duyệt vào http://localhost:5000/
   -> Nếu thấy chữ "API is running!" là thành công.

3. ⚠️ LƯU Ý QUAN TRỌNG VỀ BẢO MẬT:
   File run.py đang có sẵn một GEMINI_API_KEY được gán CỨNG (hardcode)
   trực tiếp trong code. Đây là rủi ro bảo mật nghiêm trọng — bất kỳ ai
   đọc được source code (kể cả khi đẩy lên GitHub) đều có thể lấy và
   dùng trái phép API key này.
   => Hãy:
     - Thu hồi/tạo lại (revoke & regenerate) API key đó trên Google AI
       Studio ngay nếu key đã từng bị public.
     - Sửa code để đọc key từ biến môi trường, ví dụ:
           GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
     - Thiết lập biến môi trường trước khi chạy:
         Windows (cmd):        set GEMINI_API_KEY=your_key_here
         Windows (PowerShell): $env:GEMINI_API_KEY="your_key_here"
         macOS/Linux:          export GEMINI_API_KEY=your_key_here
     - Không commit file chứa key thật lên GitHub (thêm vào .gitignore
       hoặc dùng file .env + thư viện python-dotenv).

--------------------------------------------------------------------
BƯỚC 4: CHẠY SPRING BOOT (thư mục "100 UI")
--------------------------------------------------------------------
1. Mở terminal mới, di chuyển vào thư mục "100 UI":
       cd "100 UI"

2. Kiểm tra file cấu hình application.properties (hoặc .yml), đảm bảo
   có dòng trỏ tới Python API, ví dụ:
       python.api.base-url=http://localhost:5000

3. Chạy project bằng Maven wrapper:
   - macOS/Linux:   ./mvnw spring-boot:run
   - Windows:       mvnw.cmd spring-boot:run

   Hoặc nếu đã cài Maven riêng:
       mvn spring-boot:run

   Hoặc build file jar rồi chạy:
       mvn clean package
       java -jar target/*.jar

4. Mặc định Spring Boot chạy ở: http://localhost:8080
   Mở trình duyệt vào địa chỉ này để thấy giao diện web (index.html).

--------------------------------------------------------------------
BƯỚC 5: SỬ DỤNG
--------------------------------------------------------------------
1. Đảm bảo CẢ HAI đang chạy song song:
     - Python Flask API  -> http://localhost:5000
     - Spring Boot        -> http://localhost:8080
2. Vào http://localhost:8080, upload file CSV (hoặc dùng dataset có
   sẵn qua API /process với filename tương ứng trong thư mục Dataset).
3. Đợi xử lý xong sẽ thấy dashboard: biểu đồ quy trình, Petri Net,
   BPMN, các chỉ số e-commerce, và chatbot LLM (nếu đã cấu hình đúng
   API key theo hướng dẫn ở Bước 3).

--------------------------------------------------------------------
XỬ LÝ SỰ CỐ THƯỜNG GẶP
--------------------------------------------------------------------
- Lỗi "graphviz_layout" / "Program dot not found":
    -> Chưa cài Graphviz ở hệ điều hành, hoặc chưa thêm vào PATH.

- Lỗi kết nối giữa Spring Boot và Python (Connection refused):
    -> Kiểm tra Python API đã chạy chưa, đúng cổng 5000 chưa,
       và python.api.base-url trong application.properties.

- File CSV lớn (nhiều GB) chạy chậm / tràn RAM:
    -> Điều chỉnh các biến giới hạn trong run.py:
       MAX_ROWS_FOR_PROCESS_MINING, MAX_SESSIONS_FOR_TRAINING,
       MAX_EVENT_LOG_ROWS_IN_RESPONSE — giảm giá trị để xử lý nhẹ hơn.

- Lỗi cài tensorflow trên máy cấu hình yếu / không có GPU:
    -> Dùng "tensorflow-cpu" thay cho "tensorflow" trong install.txt.
====================================================================