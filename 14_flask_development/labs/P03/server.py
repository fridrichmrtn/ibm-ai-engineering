from flask import Flask, request, jsonify, make_response, render_template
from maths.mathematics import summation, subtraction, multiplication

app = Flask("Mathematics problem solver")

@app.route("/")
def render_index_page():
    return render_template("index.html")


@app.route("/sum")
def sum_route():
    a = request.args.get("num1", type=float)
    b = request.args.get("num2", type=float)
    result = summation(a, b)
    
    if result.is_integer():
        result = int(result)
    return str(result)

@app.route("/sub")
def sub_route():
    a = request.args.get("num1", type=float)
    b = request.args.get("num2", type=float)
    result = subtraction(a, b)
    
    if result.is_integer():
        result = int(result)
    
    return str(result)

@app.route("/mul")
def mul_route():
    a = request.args.get("num1", type=float)
    b = request.args.get("num2", type=float)
    result = multiplication(a, b)

    if result.is_integer():
        result = int(result)

    return str(result)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8080)