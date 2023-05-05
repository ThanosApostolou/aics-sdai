import "../../App.css";
import Select from 'react-select';
import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";

function AddCart(props) {

  const [Customers, setCustomers] = useState([]);
  const [selectedCustomer, setSelectedCustomer] = useState("");
  const [Quantity, setQuantity] = useState("");
  const [Customer, setCustomer] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Quantity === "" || Customer === "") {
      alert("ALL the fields are mandatory!");
      return;
    }
    const cartToAdd = {
      Quantity: Quantity, Customer: Customer,
      CustomerNavigation: { Id: Customer, Username: "", Email: "", Address: "" }
    };
    props.addCartHandler(cartToAdd);
    setQuantity("");
    setCustomer("");
  };

  const handleChangeCustomer = (selectedCustomer) => {
    setSelectedCustomer(selectedCustomer);
    setCustomer(selectedCustomer.value);
  };

  const retrieveCustomers = async () => {
    const response = await api.get("/eshopUser");
    return response.data;
  }

  useEffect(() => {
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

    getAllCustomers();
  }, []);

  return (
    <div className="ui main">
      <h2>Add Cart</h2>
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
          <label>
            Quantity:
            <input
              type="text"
              name="description"
              placeholder="Description"
              value={Quantity}
              onChange={e => setQuantity(e.target.value)}
            />
          </label>
        </div>
        <button className="buttonInsert">Add</button>
      </form>
    </div>
  );

}

export default AddCart;
