import "../../App.css";
import Select from 'react-select';
import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";

function AddReview(props) {

  const [Customers, setCustomers] = useState([]);
  const [Products, setProducts] = useState([]);
  const [selectedProduct, setSelectedProduct] = useState("");
  const [selectedCustomer, setSelectedCustomer] = useState("");
  const [Customer, setCustomer] = useState("");
  const [Product, setProduct] = useState("");
  const [Rating, setRating] = useState("");
  const [Description, setDescription] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Customer === "" || Product === "" || Rating === "" || Description === "") {
      alert("ALL the fields are mandatory!");
      return;
    }
    const reviewToAdd = {
      Customer: Customer, Product: Product, Rating: Rating , Description: Description,
      CustomerNavigation: { Id: Customer, Username: "", Email: "", Address: "" },
      ProductNavigation: { Id: Product, Name: "", Description: "", Image:"", Availability:false, ProductCategory: {id:0, Name: "", Description: ""} }
    };
    props.addReviewHandler(reviewToAdd);
    setCustomer("");
    setProduct("");
    setRating("");
    setDescription("");
  };

  const handleChangeProduct = (selectedProduct) => {
    setSelectedProduct(selectedProduct);
    setProduct(selectedProduct.value);
  };

  const handleChangeCustomer = (selectedCustomer) => {
    setSelectedCustomer(selectedCustomer);
    setCustomer(selectedCustomer.value);
  };

  const retrieveProducts = async () => {
    const response = await api.get("/product");
    return response.data;
  }

  const retrieveUsers = async () => {
    const response = await api.get("/eshopUser");
    return response.data;
  }

  useEffect(() => {
    const getAllProducts = async () => {
      const allProducts = await retrieveProducts();
      if (allProducts) setProducts(
        allProducts.map((product) => {
          return {
            label: product.Name,
            value: product.Id
          }
        })
      );
    };

    const getAllUsers = async () => {
      const allUsers = await retrieveUsers();
      if (allUsers) setCustomers(
        allUsers.map((user) => {
          return {
            label: user.Username,
            value: user.Id
          }
        })
      );
    };

    getAllUsers();
    getAllProducts();
  }, []);

  return (
    <div className="ui main">
      <h2>Add Review</h2>
      <br></br>
      <form className="ui form" onSubmit={add}>
        <div className="field">
          <label>Customer</label>
          <Select
            value={selectedCustomer}
            onChange={handleChangeCustomer}
            options={Customers}
          />
        </div>
        <div className="field">
          <label>Product</label>
          <Select
            value={selectedProduct}
            onChange={handleChangeProduct}
            options={Products}
          />
        </div>
        <div className="field">
          <label>
            Rating:
            <input
              type="text"
              name="rating"
              placeholder="Rating"
              value={Rating}
              onChange={e => setRating(e.target.value)}
            />
          </label>
        </div>
        <div className="field">
          <label>
            Description:
            <input
              type="text"
              name="description"
              placeholder="Description"
              value={Description}
              onChange={e => setDescription(e.target.value)}
            />
          </label>
        </div>
        <button className="buttonInsert">Add</button>
      </form>
    </div>
  );

}

export default AddReview;
