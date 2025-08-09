from flask import Flask, request, jsonify, make_response

app = Flask(__name__)

data = [
    {
        "id": "3b58aade-8415-49dd-88db-8d7bce14932a",
        "first_name": "Tanya",
        "last_name": "Slad",
        "graduation_year": 1996,
        "address": "043 Heath Hill",
        "city": "Dayton",
        "zip": "45426",
        "country": "United States",
        "avatar": "http://dummyimage.com/139x100.png/cc0000/ffffff",
    },
    {
        "id": "d64efd92-ca8e-40da-b234-47e6403eb167",
        "first_name": "Ferdy",
        "last_name": "Garrow",
        "graduation_year": 1970,
        "address": "10 Wayridge Terrace",
        "city": "North Little Rock",
        "zip": "72199",
        "country": "United States",
        "avatar": "http://dummyimage.com/148x100.png/dddddd/000000",
    },
    {
        "id": "66c09925-589a-43b6-9a5d-d1601cf53287",
        "first_name": "Lilla",
        "last_name": "Aupol",
        "graduation_year": 1985,
        "address": "637 Carey Pass",
        "city": "Gainesville",
        "zip": "32627",
        "country": "United States",
        "avatar": "http://dummyimage.com/174x100.png/ff4444/ffffff",
    },
    {
        "id": "0dd63e57-0b5f-44bc-94ae-5c1b4947cb49",
        "first_name": "Abdel",
        "last_name": "Duke",
        "graduation_year": 1995,
        "address": "2 Lake View Point",
        "city": "Shreveport",
        "zip": "71105",
        "country": "United States",
        "avatar": "http://dummyimage.com/145x100.png/dddddd/000000",
    },
    {
        "id": "a3d8adba-4c20-495f-b4c4-f7de8b9cfb15",
        "first_name": "Corby",
        "last_name": "Tettley",
        "graduation_year": 1984,
        "address": "90329 Amoth Drive",
        "city": "Boulder",
        "zip": "80305",
        "country": "United States",
        "avatar": "http://dummyimage.com/198x100.png/cc0000/ffffff",
    }
]

@app.route("/")
def index():
    return "Welcome to the Flask server!"

@app.route("/exp")
def index_explicit():
    resp = make_response({
        "message": "Welcome to the Flask server with explicit response!"
    })
    resp.status_code = 200
    resp.headers["Content-Type"] = "application/json"
    return resp

@app.route("/no_content")
def no_content():
    return jsonify({"message": "No content available"}), 204

@app.route("/data")
def get_data():
    try:
        if data and len(data) > 0:
            return {"message":"Data of length " + str(len(data)) + " retrieved successfully", "data": data}, 200
        else:
            return {"message": "No data available"}, 404
        
    except Exception as e:
        return {"message": "An error occurred: " + str(e)}, 500
    
@app.route("/name_search", methods=["GET"])
def name_search():
    q = request.args.get("q", "")
    if not q:
        return {"message": "No search query provided"}, 400
    
    if q.strip() == "" or q.isdigit():
        return {"message": "Invalid search query"}, 422
    
    results = [item for item in data if q.lower() in item["first_name"].lower()]
    
    if results:
        return {"message": f"Found {len(results)} results", "data": results}, 200
    else:
        return {"message": "No results found"}, 404
    
@app.route("/count")
def count():
    try:
        count = len(data)
        return {"data_count": count}, 200
    except Exception as e:
        return {"message": "An error occurred while counting records: " + str(e)}, 500
    

@app.route("/person/<uuid:person_id>")
def find_by_uuid(person_id):
    try:
        person = next((item for item in data if item["id"] == str(person_id)), None)
        if person:
            return {"message": "Person found", "data": person}, 200
        else:
            return {"message": "Person not found"}, 404
    except Exception as e:
        return {"message": "An error occurred: " + str(e)}, 500
    
@app.route("/person/<uuid:person_id>", methods=["DELETE"])
def delete_person(person_id):
    try:
        global data
        data = [item for item in data if item["id"] != str(person_id)]
        return {"message": "Person deleted successfully"}, 204
    except Exception as e:
        return {"message": "An error occurred: " + str(e)}, 500
    
@app.route("/person", methods=["POST"])
def create_person():
    new_person = request.get_json()

    if not new_person or not isinstance(new_person, dict):
        return {"message": "Invalid data provided"}, 400

    required_fields = ["id", "first_name", "last_name", "graduation_year", "address", "city", "zip", "country", "avatar"]
    if not all(field in new_person for field in required_fields):
        return {"message": f"Missing fields: {', '.join(required_fields)}"}, 422
    
    # Check if the ID already exists
    if any(item["id"] == new_person["id"] for item in data):
        return {"message": "Person with this ID already exists"}, 409

    data.append(new_person)
    return {"message": "Person created successfully", "data": new_person}, 201

@app.errorhandler(404)
def not_found(error):
    return {"message": "Resource not found"}, 404

@app.errorhandler(500)
def internal_error(error):
    return {"message": "Internal server error"}, 500
