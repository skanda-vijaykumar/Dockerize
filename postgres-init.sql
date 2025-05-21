-- Create sequence for chat IDs if it doesn't exist
CREATE SEQUENCE IF NOT EXISTS chat_id_seq START WITH 1 INCREMENT BY 1 NO MAXVALUE NO CYCLE;

-- Create chat history table if it doesn't exist
CREATE TABLE IF NOT EXISTS chat_history (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    chat_id INTEGER,
    message TEXT,
    type TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on session_id for better performance
CREATE INDEX IF NOT EXISTS idx_chat_history_session_id ON chat_history(session_id);