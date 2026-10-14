import io
import pytest
from pathlib import Path
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient

from ragdoll.config import settings
from ragdoll.database import init_db
from ragdoll.main import create_app
from ragdoll.clients.embedding import EmbeddingClient
from ragdoll.clients.kev import KevClient
from ragdoll.clients.standard_reranker import StandardReranker

TEST_HEADERS = {"X-API-Key": "test-user-key"}


@pytest.fixture
def test_app(tmp_path: Path):
    db_file = tmp_path / "api_test.db"
    init_db(db_file)
    uploads_dir = tmp_path / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    settings.db_path = db_file
    settings.storage_path = uploads_dir
    settings.skip_startup_probe = True

    app = create_app()

    mock_embed = AsyncMock(spec=EmbeddingClient)
    mock_embed.check_health.return_value = True
    mock_embed.embed_single.return_value = [0.1] * 768
    mock_embed.embed_batch.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    mock_embed.embed_all.side_effect = lambda texts: [[0.1] * 768 for _ in texts]

    mock_kev = AsyncMock(spec=KevClient)
    mock_kev.check_health.return_value = True
    mock_kev.decide_boundary.return_value = False
    mock_kev.score_relevance.return_value = 1.95

    mock_standard = AsyncMock(spec=StandardReranker)
    mock_standard.check_health.return_value = True

    async def fake_rerank(query, passages):
        result = []
        for i, p in enumerate(passages):
            item = dict(p)
            item["score"] = round(0.95 - (i * 0.1), 4)
            result.append(item)
        return result

    mock_standard.rerank.side_effect = fake_rerank

    app.state.embedding_client = mock_embed
    app.state.kev_client = mock_kev
    app.state.standard_reranker = mock_standard

    return app, mock_embed, mock_kev, mock_standard


def test_auth_required_on_all_endpoints(test_app):
    app, _, _, _ = test_app
    with TestClient(app) as client:
        # 1. Health
        assert client.get("/health").status_code == 401

        # 2. Store
        file_obj = io.BytesIO(b"test data")
        assert client.post("/store", files={"file": ("test.txt", file_obj, "text/plain")}).status_code == 401

        # 3. Status
        assert client.get("/assets/some-uuid/status").status_code == 401

        # 4. Get raw asset
        assert client.get("/get/some-uuid").status_code == 401

        # 5. Search
        search_body = {"query": "test query", "top_k": 3}
        assert client.post("/search", json=search_body).status_code == 401

        # 6. Retrieve Kev
        assert client.post("/retrieve/kev", json=search_body).status_code == 401
        assert client.post("/search/kev", json=search_body).status_code == 401

        # 7. Retrieve Standard
        assert client.post("/retrieve/standard", json=search_body).status_code == 401
        assert client.post("/search/standard", json=search_body).status_code == 401

        # 8. Delete specific asset
        assert client.delete("/assets/some-uuid").status_code == 401

        # 9. Delete all assets under apikey
        assert client.delete("/assets/by-key/some-key").status_code == 401
        assert client.delete("/assets").status_code == 401


def test_auth_methods(test_app):
    app, _, _, _ = test_app
    with TestClient(app) as client:
        # X-API-Key header
        resp = client.get("/health", headers={"X-API-Key": "my-key-1"})
        assert resp.status_code == 200

        # api-key header
        resp = client.get("/health", headers={"api-key": "my-key-2"})
        assert resp.status_code == 200

        # apikey header
        resp = client.get("/health", headers={"apikey": "my-key-3"})
        assert resp.status_code == 200

        # Authorization: Bearer <key>
        resp = client.get("/health", headers={"Authorization": "Bearer my-key-4"})
        assert resp.status_code == 200

        # Query param ?api_key=<key>
        resp = client.get("/health?api_key=my-key-5")
        assert resp.status_code == 200

        # Query param ?apikey=<key>
        resp = client.get("/health?apikey=my-key-6")
        assert resp.status_code == 200


def test_health_check_ok(test_app):
    app, _, _, _ = test_app
    with TestClient(app) as client:
        resp = client.get("/health", headers=TEST_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["embeddinggemma2"] == "healthy"
        assert data["kev_4b"] == "healthy"
        assert data.get("standard_reranker") == "healthy"


def test_health_check_service_down(test_app):
    app, mock_embed, _, _ = test_app
    mock_embed.check_health.return_value = False

    with TestClient(app) as client:
        resp = client.get("/health", headers=TEST_HEADERS)
        assert resp.status_code == 503
        data = resp.json()
        assert data["status"] == "unhealthy"
        assert data["embeddinggemma2"] == "unhealthy"


def test_store_and_status_and_download(test_app):
    app, _, _, _ = test_app
    with TestClient(app) as client:
        content = b"Ragdoll test document content.\n\nSection 2 text for chunking."
        file_obj = io.BytesIO(content)

        resp = client.post(
            "/store",
            files={"file": ("report.txt", file_obj, "text/plain")},
            headers=TEST_HEADERS,
        )
        assert resp.status_code == 202
        data = resp.json()
        assert "asset_id" in data
        assert data["filename"] == "report.txt"
        assert data["size_bytes"] == len(content)
        assert data["status"] in ["queued", "coarse_indexed", "fine_indexed"]

        asset_id = data["asset_id"]

        # Deduplication test: re-upload identical file by same user
        file_obj.seek(0)
        resp2 = client.post(
            "/api/v1/assets",
            files={"file": ("report.txt", file_obj, "text/plain")},
            headers=TEST_HEADERS,
        )
        assert resp2.status_code == 202
        assert resp2.json()["asset_id"] == asset_id

        # Query status
        status_resp = client.get(f"/assets/{asset_id}/status", headers=TEST_HEADERS)
        assert status_resp.status_code == 200
        s_data = status_resp.json()
        assert s_data["asset_id"] == asset_id
        assert "status" in s_data
        assert "coarse_chunks" in s_data
        assert "fine_chunks" in s_data

        # Download original file
        get_resp = client.get(f"/get/{asset_id}", headers=TEST_HEADERS)
        assert get_resp.status_code == 200
        assert get_resp.content == content


def test_search_endpoint(test_app):
    app, _, _, _ = test_app
    with TestClient(app) as client:
        # Upload a file to populate chunks
        content = b"Financial earnings in Q3 were outstanding.\n\nOperating expenses decreased."
        client.post(
            "/store",
            files={"file": ("finances.txt", io.BytesIO(content), "text/plain")},
            headers=TEST_HEADERS,
        )

        # Search fine chunks without rerank
        search_req = {
            "query": "financial earnings Q3",
            "top_k": 3,
            "stage": "fine",
            "rerank": False,
            "include_parent_context": True,
        }
        resp = client.post("/search", json=search_req, headers=TEST_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        first = data["results"][0]
        assert "chunk_id" in first
        assert "score" in first
        assert "text" in first

        # Search with rerank enabled (default: kev)
        search_rerank = {
            "query": "financial earnings Q3",
            "top_k": 3,
            "stage": "fine",
            "rerank": True,
            "include_parent_context": True,
        }
        resp_rerank = client.post("/search", json=search_rerank, headers=TEST_HEADERS)
        assert resp_rerank.status_code == 200
        data_rerank = resp_rerank.json()
        assert len(data_rerank["results"]) >= 1
        assert data_rerank.get("rerank_method") == "kev"

        # Search with rerank enabled (rerank_method="standard")
        search_std = {
            "query": "financial earnings Q3",
            "top_k": 3,
            "stage": "fine",
            "rerank": True,
            "rerank_method": "standard",
            "include_parent_context": True,
        }
        resp_std = client.post("/search", json=search_std, headers=TEST_HEADERS)
        assert resp_std.status_code == 200
        data_std = resp_std.json()
        assert len(data_std["results"]) >= 1
        assert data_std.get("rerank_method") == "standard"


def test_retrieve_kev_endpoint(test_app):
    app, _, mock_kev, _ = test_app
    with TestClient(app) as client:
        content = b"Kev retrieval test document.\n\nParagraph 2 regarding semantic scoring."
        client.post(
            "/store",
            files={"file": ("kev_doc.txt", io.BytesIO(content), "text/plain")},
            headers=TEST_HEADERS,
        )

        req_payload = {
            "query": "semantic scoring",
            "top_k": 2,
            "stage": "fine",
            "include_parent_context": True,
        }
        # Test /retrieve/kev
        resp = client.post("/retrieve/kev", json=req_payload, headers=TEST_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        assert data["rerank_method"] == "kev"
        assert mock_kev.score_relevance.called

        # Test /search/kev alias
        resp_alias = client.post("/search/kev", json=req_payload, headers=TEST_HEADERS)
        assert resp_alias.status_code == 200
        assert resp_alias.json()["rerank_method"] == "kev"


def test_retrieve_standard_endpoint(test_app):
    app, _, _, mock_standard = test_app
    with TestClient(app) as client:
        content = b"Standard cross-encoder retrieval.\n\nSection detailing fast inference."
        client.post(
            "/store",
            files={"file": ("standard_doc.txt", io.BytesIO(content), "text/plain")},
            headers=TEST_HEADERS,
        )

        req_payload = {
            "query": "fast inference",
            "top_k": 2,
            "stage": "fine",
            "include_parent_context": True,
        }
        # Test /retrieve/standard
        resp = client.post("/retrieve/standard", json=req_payload, headers=TEST_HEADERS)
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) >= 1
        assert data["rerank_method"] == "standard"
        assert mock_standard.rerank.called

        # Test /search/standard alias
        resp_alias = client.post("/search/standard", json=req_payload, headers=TEST_HEADERS)
        assert resp_alias.status_code == 200
        assert resp_alias.json()["rerank_method"] == "standard"


def test_multi_user_api_isolation(test_app):
    app, _, _, _ = test_app
    alice_headers = {"X-API-Key": "alice-secret-key"}
    bob_headers = {"X-API-Key": "bob-secret-key"}

    with TestClient(app) as client:
        # 1. Alice uploads doc
        alice_content = b"Alice sensitive internal report.\n\nProject Moonshot details."
        alice_resp = client.post(
            "/store",
            files={"file": ("alice_doc.txt", io.BytesIO(alice_content), "text/plain")},
            headers=alice_headers,
        )
        assert alice_resp.status_code == 202
        alice_asset_id = alice_resp.json()["asset_id"]

        # 2. Bob uploads doc
        bob_content = b"Bob quarterly marketing initiatives.\n\nCampaign Solar details."
        bob_resp = client.post(
            "/store",
            files={"file": ("bob_doc.txt", io.BytesIO(bob_content), "text/plain")},
            headers=bob_headers,
        )
        assert bob_resp.status_code == 202
        bob_asset_id = bob_resp.json()["asset_id"]

        # 3. Status checks: Alice can check hers, Bob cannot check Alice's
        alice_status = client.get(f"/assets/{alice_asset_id}/status", headers=alice_headers)
        assert alice_status.status_code == 200
        assert alice_status.json()["asset_id"] == alice_asset_id

        bob_trying_alice_status = client.get(f"/assets/{alice_asset_id}/status", headers=bob_headers)
        assert bob_trying_alice_status.status_code == 404

        # 4. Download checks: Alice can download hers, Bob cannot download Alice's
        alice_download = client.get(f"/get/{alice_asset_id}", headers=alice_headers)
        assert alice_download.status_code == 200
        assert alice_download.content == alice_content

        bob_trying_alice_download = client.get(f"/get/{alice_asset_id}", headers=bob_headers)
        assert bob_trying_alice_download.status_code == 404

        # 5. Search checks: Alice only finds Alice's content; Bob only finds Bob's content
        search_payload = {
            "query": "details",
            "top_k": 10,
            "stage": "fine",
            "rerank": False,
        }
        alice_search = client.post("/search", json=search_payload, headers=alice_headers)
        assert alice_search.status_code == 200
        alice_results = alice_search.json()["results"]
        assert all(r["asset_id"] == alice_asset_id for r in alice_results)

        bob_search = client.post("/search", json=search_payload, headers=bob_headers)
        assert bob_search.status_code == 200
        bob_results = bob_search.json()["results"]
        assert all(r["asset_id"] == bob_asset_id for r in bob_results)

        # 6. Deduplication isolation: Bob uploading identical content as Alice gets a new asset
        bob_same_as_alice = client.post(
            "/store",
            files={"file": ("same.txt", io.BytesIO(alice_content), "text/plain")},
            headers=bob_headers,
        )
        assert bob_same_as_alice.status_code == 202
        # Must NOT be Alice's asset id
        assert bob_same_as_alice.json()["asset_id"] != alice_asset_id


def test_delete_specific_asset_endpoint(test_app):
    app, _, _, _ = test_app
    alice_headers = {"X-API-Key": "alice-secret-key"}
    bob_headers = {"X-API-Key": "bob-secret-key"}

    with TestClient(app) as client:
        # 1. Alice uploads doc
        content = b"Content to be deleted.\n\nExtra information."
        resp = client.post(
            "/store",
            files={"file": ("to_delete.txt", io.BytesIO(content), "text/plain")},
            headers=alice_headers,
        )
        assert resp.status_code == 202
        asset_id = resp.json()["asset_id"]

        # Verify asset exists
        status_resp = client.get(f"/assets/{asset_id}/status", headers=alice_headers)
        assert status_resp.status_code == 200

        # Bob cannot delete Alice's asset
        bob_del = client.delete(f"/assets/{asset_id}", headers=bob_headers)
        assert bob_del.status_code == 404

        # Alice deletes her asset
        del_resp = client.delete(f"/assets/{asset_id}", headers=alice_headers)
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert del_data["deleted"] is True
        assert del_data["asset_id"] == asset_id

        # Verify asset is no longer accessible
        assert client.get(f"/assets/{asset_id}/status", headers=alice_headers).status_code == 404
        assert client.get(f"/get/{asset_id}", headers=alice_headers).status_code == 404

        # Search should not find deleted asset
        search_resp = client.post(
            "/search",
            json={"query": "deleted", "top_k": 5},
            headers=alice_headers,
        )
        assert search_resp.status_code == 200
        assert len(search_resp.json()["results"]) == 0

        # Repeating delete returns 404
        assert client.delete(f"/assets/{asset_id}", headers=alice_headers).status_code == 404

        # Re-uploading same content works without sha256 conflict
        reupload_resp = client.post(
            "/store",
            files={"file": ("to_delete.txt", io.BytesIO(content), "text/plain")},
            headers=alice_headers,
        )
        assert reupload_resp.status_code == 202
        new_asset_id = reupload_resp.json()["asset_id"]
        assert new_asset_id != asset_id


def test_delete_all_assets_under_apikey_endpoint(test_app):
    app, _, _, _ = test_app
    alice_headers = {"X-API-Key": "alice-key-123"}
    bob_headers = {"X-API-Key": "bob-key-456"}

    with TestClient(app) as client:
        # Alice uploads 2 docs
        resp1 = client.post(
            "/store",
            files={"file": ("doc1.txt", io.BytesIO(b"Doc 1 content"), "text/plain")},
            headers=alice_headers,
        )
        assert resp1.status_code == 202
        id1 = resp1.json()["asset_id"]

        resp2 = client.post(
            "/store",
            files={"file": ("doc2.txt", io.BytesIO(b"Doc 2 content"), "text/plain")},
            headers=alice_headers,
        )
        assert resp2.status_code == 202
        id2 = resp2.json()["asset_id"]

        # Bob uploads 1 doc
        resp_bob = client.post(
            "/store",
            files={"file": ("bob.txt", io.BytesIO(b"Bob content"), "text/plain")},
            headers=bob_headers,
        )
        assert resp_bob.status_code == 202
        bob_id = resp_bob.json()["asset_id"]

        # Verify all 3 exist
        assert client.get(f"/assets/{id1}/status", headers=alice_headers).status_code == 200
        assert client.get(f"/assets/{id2}/status", headers=alice_headers).status_code == 200
        assert client.get(f"/assets/{bob_id}/status", headers=bob_headers).status_code == 200

        # Delete all Alice assets using path endpoint
        del_resp = client.delete("/assets/by-key/alice-key-123", headers=alice_headers)
        assert del_resp.status_code == 200
        del_data = del_resp.json()
        assert del_data["deleted_count"] == 2
        assert del_data["apikey"] == "alice-key-123"
        assert set(del_data["asset_ids"]) == {id1, id2}

        # Alice's assets are gone
        assert client.get(f"/assets/{id1}/status", headers=alice_headers).status_code == 404
        assert client.get(f"/assets/{id2}/status", headers=alice_headers).status_code == 404

        # Bob's asset is still intact
        assert client.get(f"/assets/{bob_id}/status", headers=bob_headers).status_code == 200

        # Delete Bob's asset using query param endpoint DELETE /assets?target_apikey=bob-key-456
        del_bob = client.delete("/assets?target_apikey=bob-key-456", headers=bob_headers)
        assert del_bob.status_code == 200
        assert del_bob.json()["deleted_count"] == 1
        assert del_bob.json()["asset_ids"] == [bob_id]

        assert client.get(f"/assets/{bob_id}/status", headers=bob_headers).status_code == 404

        # Deleting again returns 0 deleted count
        repeat_resp = client.delete("/assets", headers=bob_headers)
        assert repeat_resp.status_code == 200
        assert repeat_resp.json()["deleted_count"] == 0
