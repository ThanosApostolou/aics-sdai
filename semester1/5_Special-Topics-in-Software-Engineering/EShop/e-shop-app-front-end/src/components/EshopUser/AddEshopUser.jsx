import "../../App.css";
import React, { useState, useEffect } from 'react';

function AddEshopUser(props) {
  const [Username, setUsername] = useState("");
  const [Email, setEmail] = useState("");
  const [Address, setAddress] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Username === "" || Email === "" || Address === "") {
      alert("ALl the fields are mandatory!");
      return;
    }
    
    const eShopUserToAdd = {
      Username: Username, Email: Email, Address:Address
    };

    props.addEshopUserHandler(eShopUserToAdd);
    setUsername("");
    setEmail("");
    setAddress("");
  };
  
    return (
      <div className="ui main">
        <h2>Add User</h2>
        <form className="ui form" onSubmit={add}>
          <div className="field">
            <label>Username</label>
            <input
              type="text"
              name="Username"
              placeholder="Username"
              value={Username}
              onChange={e => setUsername(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Email</label>
            <input
              type="text"
              name="Email"
              placeholder="Email"
              value={Email}
              onChange={e => setEmail(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Address</label>
            <input
              type="text"
              name="Address"
              placeholder="Address"
              value={Address}
              onChange={e => setAddress(e.target.value)}
            />
          </div>
          <button className="buttonInsert">Add</button>
        </form>
      </div>
    );
}

export default AddEshopUser;
