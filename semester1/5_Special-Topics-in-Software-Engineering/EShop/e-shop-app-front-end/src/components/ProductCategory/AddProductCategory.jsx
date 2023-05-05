import "../../App.css";
import React, { useState, useEffect } from 'react';

function AddProductCategory(props) {
  const [Name, setName] = useState("");
  const [Description, setDescription] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Name === "" || Description === "") {
      alert("ALl the fields are mandatory!");
      return;
    }
    
    const productCategoryToAdd = {
      Name: Name, Description: Description
    };

    props.addProductCategoryHandler(productCategoryToAdd);
    setName("");
    setDescription("");
  };
  
    return (
      <div className="ui main">
        <h2>Add Product</h2>
        <form className="ui form" onSubmit={add}>
          <div className="field">
            <label>Name</label>
            <input
              type="text"
              name="Name"
              placeholder="Name"
              value={Name}
              onChange={e => setName(e.target.value)}
            />
          </div>
          <div className="field">
            <label>Description</label>
            <input
              type="text"
              name="description"
              placeholder="Description"
              value={Description}
              onChange={e => setDescription(e.target.value)}
            />
          </div>
          <button className="buttonInsert">Add</button>
        </form>
      </div>
    );
}

export default AddProductCategory;
