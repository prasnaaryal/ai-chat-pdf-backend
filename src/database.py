import psycopg2

class DatabaseManager:
    def __init__(self, db_config):
        self.db_config = db_config
        self.connection = psycopg2.connect(
            dbname=db_config['database'],
            host=db_config['host'],
            user=db_config['user'],
            password=db_config['password'],
            port=db_config['port']
        )
        self.connection.autocommit = True

    def close(self):
        self.connection.close()

    def execute_query(self, query):
        with self.connection.cursor() as cursor:
            cursor.execute(query)
            try:
                return cursor.fetchall()
            except psycopg2.ProgrammingError:
                return None  # No results to fetch


    def get_vector_store(self):
        from sqlalchemy import make_url
        from llama_index.vector_stores.postgres import PGVectorStore

        vector_store = PGVectorStore.from_params(
            database=self.db_config['database'],
            host=self.db_config['host'],
            user=self.db_config['user'],
            password=self.db_config['password'],
            port=self.db_config['port'],
            table_name="llama2_pdf",
            embed_dim=384,  # openai embedding dimension
        )

        return vector_store
