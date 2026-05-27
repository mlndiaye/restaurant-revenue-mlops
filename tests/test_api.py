from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_check():
    """Test that the /health endpoint is working and returns the expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    # Status can be healthy (model loaded) or degraded (no model loaded, e.g. in test envs)
    assert data["status"] in ["healthy", "degraded"]


def test_predict_single_restaurant():
    """Test that predicting a single restaurant works and returns a valid float prediction."""
    payload = {
        "Open Date": "01/01/2010",
        "City Group": "Big Cities",
        "Type": "IL",
        "P1": 4.0, "P2": 5.0, "P3": 4.0, "P4": 4.0, "P5": 2.0, "P6": 2.0, "P7": 5.0,
        "P8": 4.0, "P9": 5.0, "P10": 5.0, "P11": 3.0, "P12": 5.0, "P13": 5.0, "P14": 1.0,
        "P15": 2.0, "P16": 2.0, "P17": 2.0, "P18": 4.0, "P19": 5.0, "P20": 4.0, "P21": 1.0,
        "P22": 3.0, "P23": 3.0, "P24": 1.0, "P25": 1.0, "P26": 1.0, "P27": 4.0, "P28": 2.0,
        "P29": 3.0, "P30": 5.0, "P31": 3.0, "P32": 4.0, "P33": 5.0, "P34": 5.0, "P35": 4.0,
        "P36": 3.0, "P37": 4.0
    }
    
    # We trigger the startup event manually so the TestClient loads the model
    with TestClient(app) as test_client:
        response = test_client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], float)
        assert data["prediction"] > 0


def test_predict_invalid_data():
    """Test that missing required fields throws a 422 validation error."""
    # Missing required field "Type" (restaurant_type)
    payload = {
        "Open Date": "01/01/2010",
        "City Group": "Big Cities",
        "P1": 4.0, "P2": 5.0, "P3": 4.0, "P4": 4.0, "P5": 2.0, "P6": 2.0, "P7": 5.0,
        "P8": 4.0, "P9": 5.0, "P10": 5.0, "P11": 3.0, "P12": 5.0, "P13": 5.0, "P14": 1.0,
        "P15": 2.0, "P16": 2.0, "P17": 2.0, "P18": 4.0, "P19": 5.0, "P20": 4.0, "P21": 1.0,
        "P22": 3.0, "P23": 3.0, "P24": 1.0, "P25": 1.0, "P26": 1.0, "P27": 4.0, "P28": 2.0,
        "P29": 3.0, "P30": 5.0, "P31": 3.0, "P32": 4.0, "P33": 5.0, "P34": 5.0, "P35": 4.0,
        "P36": 3.0, "P37": 4.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error code
