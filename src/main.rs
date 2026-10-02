use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::get,
    Router,
};
use sqlx::{PgPool, Row};
use tower_http::services::{ServeDir, ServeFile};

async fn get_brain(State(pool): State<PgPool>) -> impl axum::response::IntoResponse {
    let row = sqlx::query("SELECT weights FROM brain WHERE id = 1")
        .fetch_optional(&pool)
        .await;
    match row {
        Ok(Some(r)) => {
            let weights: String = r.get("weights");
            (StatusCode::OK, weights).into_response()
        }
        _ => StatusCode::NO_CONTENT.into_response(),
    }
}

async fn put_brain(State(pool): State<PgPool>, weights: String) -> StatusCode {
    let res = sqlx::query(
        "INSERT INTO brain (id, weights, updated_at) VALUES (1, $1, now())
         ON CONFLICT (id) DO UPDATE SET weights = $1, updated_at = now()",
    )
    .bind(&weights)
    .execute(&pool)
    .await;
    match res {
        Ok(_) => StatusCode::OK,
        Err(e) => {
            eprintln!("put_brain error: {e}");
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

#[tokio::main]
async fn main() {
    let db_url = std::env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    let pool = PgPool::connect(&db_url).await.expect("connect to postgres");
    sqlx::query(
        "CREATE TABLE IF NOT EXISTS brain (id INT PRIMARY KEY, weights TEXT NOT NULL, updated_at TIMESTAMPTZ NOT NULL DEFAULT now())",
    )
    .execute(&pool)
    .await
    .expect("create table");

    let dist = std::env::var("DIST_DIR").unwrap_or_else(|_| "frontend/dist".into());
    let app = Router::new()
        .route("/api/brain", get(get_brain).put(put_brain))
        .fallback_service(
            ServeDir::new(&dist).fallback(ServeFile::new(format!("{dist}/index.html"))),
        )
        .with_state(pool);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(8080);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.unwrap();
    println!("listening on 0.0.0.0:{port}");
    axum::serve(listener, app).await.unwrap();
}
