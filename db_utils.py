from typing import Dict, Any, Optional, List
import mysql.connector
from mysql.connector import Error
from mysql.connector.connection import MySQLConnection
from mysql.connector.cursor import MySQLCursor
import logging
from contextlib import contextmanager
from dataclasses import dataclass
import os
from dotenv import load_dotenv

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DocumentData:
    """문서 데이터를 위한 데이터 클래스"""
    file_path: str
    content: str
    document_type: str
    processed_date: str
    metadata: Dict[str, Any] = None

class DatabaseManager:
    def __init__(self, config_path: str = None):
        """데이터베이스 매니저 초기화"""
        load_dotenv()  # .env 파일에서 환경 변수 로드
        self.config = self._load_config(config_path)
        self.connection = None
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, str]:
        """설정 로드 (환경 변수 또는 설정 파일에서)"""
        if config_path:
            # 설정 파일에서 로드하는 로직 구현
            pass
        
        return {
            'host': os.getenv('DB_HOST', 'localhost'),
            'user': os.getenv('DB_USER', 'root'),
            'password': os.getenv('DB_PASSWORD', ''),
            'database': os.getenv('DB_NAME', 'saef_db'),
            'port': os.getenv('DB_PORT', '3306')
        }

    @contextmanager
    def get_connection(self) -> MySQLConnection:
        """데이터베이스 연결을 컨텍스트 매니저로 제공"""
        try:
            if not self.connection or not self.connection.is_connected():
                self.connection = mysql.connector.connect(**self.config)
            yield self.connection
        except Error as e:
            logger.error(f"데이터베이스 연결 오류: {e}")
            raise
        finally:
            if self.connection and self.connection.is_connected():
                self.connection.close()

    @contextmanager
    def get_cursor(self) -> MySQLCursor:
        """데이터베이스 커서를 컨텍스트 매니저로 제공"""
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=True)
            try:
                yield cursor
                connection.commit()
            except Error as e:
                connection.rollback()
                logger.error(f"쿼리 실행 오류: {e}")
                raise
            finally:
                cursor.close()

    def create_tables(self) -> None:
        """필요한 테이블들을 생성"""
        create_documents_table = """
        CREATE TABLE IF NOT EXISTS documents (
            id INT AUTO_INCREMENT PRIMARY KEY,
            file_path VARCHAR(255) NOT NULL,
            content TEXT NOT NULL,
            document_type VARCHAR(50),
            processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            UNIQUE KEY unique_file_path (file_path)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        
        create_embeddings_table = """
        CREATE TABLE IF NOT EXISTS embeddings (
            id INT AUTO_INCREMENT PRIMARY KEY,
            document_id INT NOT NULL,
            embedding_vector BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """

        with self.get_cursor() as cursor:
            cursor.execute(create_documents_table)
            cursor.execute(create_embeddings_table)
            logger.info("테이블이 성공적으로 생성되었습니다.")

    def insert_document(self, document: DocumentData) -> int:
        """문서 데이터를 데이터베이스에 삽입"""
        insert_query = """
        INSERT INTO documents (file_path, content, document_type, processed_date)
        VALUES (%(file_path)s, %(content)s, %(document_type)s, %(processed_date)s)
        ON DUPLICATE KEY UPDATE
            content = VALUES(content),
            document_type = VALUES(document_type),
            processed_date = VALUES(processed_date)
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(insert_query, document.__dict__)
            return cursor.lastrowid

    def insert_embedding(self, document_id: int, embedding_vector: bytes) -> None:
        """문서의 임베딩 벡터를 데이터베이스에 저장"""
        insert_query = """
        INSERT INTO embeddings (document_id, embedding_vector)
        VALUES (%s, %s)
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(insert_query, (document_id, embedding_vector))

    def get_document_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
        """ID로 문서 조회"""
        select_query = "SELECT * FROM documents WHERE id = %s"
        
        with self.get_cursor() as cursor:
            cursor.execute(select_query, (document_id,))
            return cursor.fetchone()

    def get_documents_by_type(self, document_type: str) -> List[Dict[str, Any]]:
        """문서 타입으로 문서들 조회"""
        select_query = "SELECT * FROM documents WHERE document_type = %s"
        
        with self.get_cursor() as cursor:
            cursor.execute(select_query, (document_type,))
            return cursor.fetchall()

    def search_documents(self, query: str) -> List[Dict[str, Any]]:
        """전체 텍스트 검색으로 문서 검색"""
        search_query = """
        SELECT * FROM documents 
        WHERE MATCH(content) AGAINST(%s IN NATURAL LANGUAGE MODE)
        """
        
        with self.get_cursor() as cursor:
            cursor.execute(search_query, (query,))
            return cursor.fetchall()

    def cleanup(self) -> None:
        """연결 정리 및 리소스 해제"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("데이터베이스 연결이 종료되었습니다.")

# 사용 예시
if __name__ == "__main__":
    try:
        # 데이터베이스 매니저 초기화
        db_manager = DatabaseManager()
        
        # 테이블 생성
        db_manager.create_tables()
        
        # 샘플 문서 삽입
        sample_doc = DocumentData(
            file_path="/path/to/document.pdf",
            content="샘플 문서 내용",
            document_type="pdf",
            processed_date="2024-01-01 00:00:00"
        )
        
        doc_id = db_manager.insert_document(sample_doc)
        logger.info(f"문서가 성공적으로 삽입되었습니다. ID: {doc_id}")
        
        # 문서 조회
        document = db_manager.get_document_by_id(doc_id)
        logger.info(f"조회된 문서: {document}")
        
    except Exception as e:
        logger.error(f"에러 발생: {e}")
    finally:
        db_manager.cleanup()
