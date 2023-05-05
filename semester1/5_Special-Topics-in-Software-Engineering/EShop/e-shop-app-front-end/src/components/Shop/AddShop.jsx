import "../../App.css";
import Select from 'react-select';
import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";

function AddShop(props) {

  const [ShopCategories, setShopCategories] = useState([]);
  const [Sellers, setSellers] = useState([]);
  const [selectedShopCategory, setSelectedShopCategory] = useState("");
  const [selectedSeller, setSelectedSeller] = useState("");
  const [Name, setName] = useState("");
  const [Image, setImage] = useState("");
  const [ShopCategory, setShopCategory] = useState("");
  const [Seller, setSeller] = useState("");
  const [Description, setDescription] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Name === "" || ShopCategory === "" || Image === "" || Description === "") {
      alert("ALL the fields are mandatory!");
      return;
    }
    const shopToAdd = {
      Name: Name, Seller: Seller, ShopCategory: ShopCategory, Image: Image, Description: Description,
      ShopCategoryNavigation: { Id: ShopCategory, Name: "", Description: "" },
      SellerNavigation: { Id: Seller, Username: "", Email: "", Address: "" }
    };
    props.addShopHandler(shopToAdd);
    setName("");
    setSeller("");
    setShopCategory("");
    setImage("");
    setDescription("");
  };

  const handleChangeShopCategory = (selectedShopCategory) => {
    setSelectedShopCategory(selectedShopCategory);
    setShopCategory(selectedShopCategory.value);
  };

  const handleChangeSeller = (selectedSeller) => {
    setSelectedSeller(selectedSeller);
    setSeller(selectedSeller.value);
  };

  const retrieveShopCategories = async () => {
    const response = await api.get("/shopCategory");
    return response.data;
  }

  const retrieveSellers = async () => {
    const response = await api.get("/eshopUser");
    return response.data;
  }

  useEffect(() => {
    const getAllShopCategories = async () => {
      const allShopCategories = await retrieveShopCategories();
      if (allShopCategories) setShopCategories(
        allShopCategories.map((shopCat) => {
          return {
            label: shopCat.Name,
            value: shopCat.Id
          }
        })
      );
    };

    const getAllSellers = async () => {
      const allSellers = await retrieveSellers();
      if (allSellers) setSellers(
        allSellers.map((seller) => {
          return {
            label: seller.Username,
            value: seller.Id
          }
        })
      );
    };

    getAllSellers();
    getAllShopCategories();
  }, []);

  return (
    <div className="ui main">
      <h2>Add Shop</h2>
      <br></br>
      <form className="ui form" onSubmit={add}>
        <div className="field">
          <label>Seller</label>
          <Select
            value={selectedSeller}
            onChange={handleChangeSeller}
            options={Sellers}
          />
        </div>
        <div className="field">
          <label>ShopCategory</label>
          <Select
            value={selectedShopCategory}
            onChange={handleChangeShopCategory}
            options={ShopCategories}
          />
        </div>
        <div className="field">
          <label>
            Shop Name:
            <input
              type="text"
              name="Name"
              placeholder="Name"
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
        <button className="buttonInsert">Add</button>
      </form>
    </div>
  );

}

export default AddShop;
