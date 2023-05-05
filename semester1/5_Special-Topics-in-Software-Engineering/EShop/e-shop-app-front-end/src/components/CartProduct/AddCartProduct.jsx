import "../../App.css";
import Select from 'react-select';
import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";

function AddCartProduct(props) {

  const [Carts, setCarts] = useState([]);
  const [Products, setProducts] = useState([]);
  const [selectedCart, setSelectedCart] = useState("");
  const [selectedProduct, setSelectedProduct] = useState("");
  const [Cart, setCart] = useState("");
  const [Product, setProduct] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Cart === "" || Product === "") {
      alert("ALL the fields are mandatory!");
      return;
    }
    const cartProductToAdd = {
      Cart: Cart,
      Product: Product, 
      CartNavigation: { Id: Cart, Quantity: 0, Customer: 1,CustomerNavigation:{id:1, Username: "", Email: "", Address: "" } },
      ProductNavigation: { Id: Product, Name: "", Description: "", Image:"", Availability:false,ProductCategory:{Id: 1, Name: "", Description: ""} }
    };
    props.addCartProductHandler(cartProductToAdd);
    setCart("");
    setProduct("");
  };

  const handleChangeCart = (selectedCart) => {
    setSelectedCart(selectedCart);
    setCart(selectedCart.value);
  };

  const handleChangeProduct = (selectedProduct) => {
    setSelectedProduct(selectedProduct);
    setProduct(selectedProduct.value);
  };

  const retrieveCarts = async () => {
    const response = await api.get("/cart");
    return response.data;
  }

  const retrieveProducts = async () => {
    const response = await api.get("/product");
    return response.data;
  }

  useEffect(() => {
    const getAllCarts = async () => {
      const allCarts = await retrieveCarts();
      if (allCarts) setCarts(
        allCarts.map((cart) => {
          return {
            label: cart.Id,
            value: cart.Id
          }
        })
      );
    };

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

    getAllCarts();
    getAllProducts();
  }, []);

  return (
    <div className="ui main">
      <h2>Add Cart with Products</h2>
      <br></br>
      <form className="ui form" onSubmit={add}>
        <div className="field">
          <label>Cart</label>
          <Select
            value={selectedCart}
            onChange={handleChangeCart}
            options={Carts}
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
        <button className="buttonInsert">Add</button>
      </form>
    </div>
  );

}

export default AddCartProduct;
