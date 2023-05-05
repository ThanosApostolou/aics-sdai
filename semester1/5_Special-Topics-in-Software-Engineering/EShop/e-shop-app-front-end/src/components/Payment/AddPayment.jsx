import "../../App.css";
import Select from 'react-select';
import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";

function AddPayment(props) {

  const [PaymentCategories, setPaymentCategories] = useState([]);
  const [selectedPaymentCategory, setSelectedPaymentCategory] = useState("");
  const [Availability, setAvailability] = useState("");
  const [Amount, setAmount] = useState("");
  const [PaymentCategoryId, setPaymentCategoryId] = useState("");

  const add = (e) => {
    e.preventDefault();
    if (Availability === "" || Amount === "" || PaymentCategoryId === "") {
      alert("ALL the fields are mandatory!");
      return;
    }
    const paymentToAdd = {
      Availability: Availability, Amount: Amount, PaymentCategoryId: PaymentCategoryId,
      PaymentCategory: { Id: PaymentCategoryId, Name: "", Description: "" }
    };
    props.addPaymentHandler(paymentToAdd);
    setAvailability("");
    setAmount("");
    setPaymentCategoryId("");
  };

  const handleChangePaymentCategory = (selectedPaymentCategory) => {
    setSelectedPaymentCategory(selectedPaymentCategory);
    setPaymentCategoryId(selectedPaymentCategory.value);
  };

  const retrievePaymentCategories = async () => {
    const response = await api.get("/paymentCategory");
    return response.data;
  }

  useEffect(() => {
    const getAllPaymentCategories = async () => {
      const allPaymentCategories = await retrievePaymentCategories();
      if (allPaymentCategories) setPaymentCategories(
        allPaymentCategories.map((paymentCat) => {
          return {
            label: paymentCat.Name,
            value: paymentCat.Id
          }
        })
      );
    };

    getAllPaymentCategories();
  }, []);

  return (
    <div className="ui main">
      <h2>Add Payment</h2>
      <br></br>
      <form className="ui form" onSubmit={add}>
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
        <div className="field">
          <label>Payment Category</label>
          <Select
            value={selectedPaymentCategory}
            onChange={handleChangePaymentCategory}
            options={PaymentCategories}
          />
        </div>
        <div className="field">
          <label>
            Amount:
            <input
              type="text"
              name="Amount"
              placeholder="Amount Name"
              value={Amount}
              onChange={e => setAmount(e.target.value)}
            />
          </label>
        </div>
        <button className="buttonInsert">Add</button>
      </form>
    </div>
  );

}

export default AddPayment;
