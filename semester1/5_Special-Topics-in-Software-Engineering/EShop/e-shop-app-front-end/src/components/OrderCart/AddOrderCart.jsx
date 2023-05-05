import "../../App.css";
import Select from 'react-select';
import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";

function AddOrderCart(props) {

  const [Carts, setCarts] = useState([]);
  const [Customers, setCustomers] = useState([]);
  const [Payments, setPayments] = useState([]);
  const [selectedCart, setSelectedCart] = useState("");
  const [selectedCustomer, setSelectedCustomer] = useState("");
  const [selectedPayment, setSelectedPayment] = useState("");
  const [Cart, setCart] = useState("");
  const [Customer, setCustomer] = useState("");
  const [Payment, setPayment] = useState("");
  const [DeliveryAdress, setDeliveryAdress] = useState("");
  const [Date, setDate] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Cart === "" || Customer === "" || Payment === "") {
      alert("ALL the fields are mandatory!");
      return;
    }
    const orderCartToAdd = {
      DeliveryAdress :DeliveryAdress,
      Date :Date,
      Cart: Cart,
      Customer: Customer, 
      Payment: Payment, 
      CartNavigation: { Id: Cart, Quantity: 0, Customer: 1,CustomerNavigation:{id:1, Username: "", Email: "", Address: "" } },
      CustomerNavigation: { Id: Customer, Username: "", Email: "", Address: "" },
      PaymentNavigation: { Id: Payment, Availability: false, Amount: 0, PaymentCategoryId: 0,
        PaymentCategory: { Id: 0, Name: "", Description: "" } }
    };
    props.addOrderCartHandler(orderCartToAdd);
    setDeliveryAdress("");
    setDate("");
    setCart("");
    setCustomer("");
    setPayment("");
  };

  const handleChangeCart = (selectedCart) => {
    setSelectedCart(selectedCart);
    setCart(selectedCart.value);
  };

  const handleChangeCustomer = (selectedCustomer) => {
    setSelectedCustomer(selectedCustomer);
    setCustomer(selectedCustomer.value);
  };

  const handleChangePayment = (selectedPayment) => {
    setSelectedPayment(selectedPayment);
    setPayment(selectedPayment.value);
  };

  const retrieveCarts = async () => {
    const response = await api.get("/cart");
    return response.data;
  }

  const retrieveCustomers = async () => {
    const response = await api.get("/eshopUser");
    return response.data;
  }

  const retrievePayments = async () => {
    const response = await api.get("/payment");
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

    const getAllCustomers = async () => {
      const allCustomers = await retrieveCustomers();
      if (allCustomers) setCustomers(
        allCustomers.map((customer) => {
          return {
            label: customer.Username,
            value: customer.Id
          }
        })
      );
    };

    const getAllPayments = async () => {
      const allPayments = await retrievePayments();
      if (allPayments) setPayments(
        allPayments.map((payment) => {
          return {
            label: payment.Id,
            value: payment.Id
          }
        })
      );
    };

    getAllCarts();
    getAllCustomers();
    getAllPayments();
  }, []);

  return (
    <div className="ui main">
      <h2>Add Order</h2>
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
          <label>Customer</label>
          <Select
            value={selectedCustomer}
            onChange={handleChangeCustomer}
            options={Customers}
          />
        </div>
        <div className="field">
          <label>Payment</label>
          <Select
            value={selectedPayment}
            onChange={handleChangePayment}
            options={Payments}
          />
        </div>
          <div className="field">
            <label>Delivery Adress</label>
            <input
              type="text"
              name="Address"
              placeholder="Address"
              value={DeliveryAdress}
              onChange={e => setDeliveryAdress(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Date</label>
            <input
              type="date"
              name="Date"
              placeholder="Date"
              value={Date}
              onChange={e => setDate(e.target.value)}
            />
          </div>
        <button className="buttonInsert">Add</button>
      </form>
    </div>
  );

}

export default AddOrderCart;
