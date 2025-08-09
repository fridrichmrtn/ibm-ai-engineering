# Import libraries
from flask import Flask, jsonify, request, render_template, redirect, url_for

# Instantiate Flask functionality
app = Flask("My Flask App")

# Sample data
transactions = [
    {"id": 1, "date": "2023-06-01", "amount": 100},
    {"id": 2, "date": "2023-06-02", "amount": -200},
    {"id": 3, "date": "2023-06-03", "amount": 300}
]
# Read operation
@app.route("/", methods=["GET"])
def get_transactions():
    return render_template("transactions.html", transactions=transactions)

# Create operation
@app.route("/add", methods=["POST", "GET"])
def add_transaction():
    
    if request.method == "POST":

        new_transaction = {
            "id": len(transactions) + 1,
            "date": request.form["date"],
            "amount": float(request.form["amount"])
        }

        transactions.append(new_transaction)
        return redirect(url_for("get_transactions"))
    return render_template("form.html")

# Update operation
@app.route("/edit/<int:transaction_id>", methods=["POST", "GET"])
def edit_transaction(transaction_id):
    if request.method == "POST":
        date = request.form["date"]
        amount = float(request.form["amount"])

        for transaction in transactions:
            if transaction["id"] == transaction_id:
                transaction["date"] = date
                transaction["amount"] = amount
                break

        return redirect(url_for("get_transactions"))

    for transaction in transactions:
        if transaction["id"] == transaction_id:
            return render_template("edit.html", transaction=transaction)

# Delete operation
@app.route("/delete/<int:transaction_id>")
def delete_transaction(transaction_id):
    global transactions
    transactions = [t for t in transactions if t["id"] != transaction_id]
    return redirect(url_for("get_transactions"))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
    