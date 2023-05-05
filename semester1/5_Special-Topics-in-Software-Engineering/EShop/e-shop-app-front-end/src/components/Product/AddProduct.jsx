import "../../App.css";
import Select from 'react-select';
import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";

function AddProduct(props) {

  const [ProductCategories, setProductCategories] = useState([]);
  const [selectedProductCategory, setSelectedProductCategory] = useState("");
  const [Name, setName] = useState("");
  const [Description, setDescription] = useState("");
  const [Image, setImage] = useState("");
  const [Availability, setAvailability] = useState("");
  const [ProductCategoryId, setProductCategoryId] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Name === "" || Description === "" || Image === "" || Availability === "" || ProductCategoryId === "") {
      alert("ALL the fields are mandatory!");
      return;
    }
    const productToAdd = {
      Name: Name, Description: Description, Image: Image, Availability: Availability, ProductCategoryId: ProductCategoryId,
      ProductCategory: { Id: ProductCategoryId, Name: "", Description: "" }
    };
    props.addProductHandler(productToAdd);
    setName("");
    setDescription("");
    setImage("");
    setAvailability("");
    setProductCategoryId("");
  };

  const handleChangeProductCategory = (selectedProductCategory) => {
    setSelectedProductCategory(selectedProductCategory);
    setProductCategoryId(selectedProductCategory.value);
  };

  const retrieveProductCategories = async () => {
    const response = await api.get("/productCategory");
    return response.data;
  }

  useEffect(() => {
    const getAllProductCategories = async () => {
      const allProductCategories = await retrieveProductCategories();
      if (allProductCategories) setProductCategories(
        allProductCategories.map((productCat) => {
          return {
            label: productCat.Name,
            value: productCat.Id
          }
        })
      );
    };

    getAllProductCategories();
  }, []);

  return (
    <div className="ui main">
      <h2>Add Product</h2>
      <br></br>
      <form className="ui form" onSubmit={add}>
        <div className="field">
          <label>Product Category</label>
          <Select
            value={selectedProductCategory}
            onChange={handleChangeProductCategory}
            options={ProductCategories}
          />
        </div>
        <div className="field">
          <label>
            Product Name:
            <input
              type="text"
              name="Name"
              placeholder="Product Name"
              value={Name}
              onChange={e => setName(e.target.value)}
            />
          </label>
        </div>
        <div className="field">
          <label>
            Image:
            <input
              type="text"
              name="image"
              placeholder="Image"
              value={Image}
              onChange={e => setImage(e.target.value)}
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
        <div className="field">
          <label>
            Availability:
            <input
              type="text"
              name="availability"
              placeholder="Availability"
              value={Availability}
              onChange={e => setAvailability(true)}
            />
          </label>
        </div>
        <button className="buttonInsert">Add</button>
      </form>
    </div>
  );

}

export default AddProduct;
