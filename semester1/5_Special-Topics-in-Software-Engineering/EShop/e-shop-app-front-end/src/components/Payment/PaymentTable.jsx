import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import "../../App.css";
import AddPaymentModal from "./AddPaymentModal";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";
import Select from 'react-select';

function PaymentDataDisplay() {
    const [PaymentCategories, setPaymentCategories] = useState([]);
    const [selectedPaymentCategory, setselectedPaymentCategory] = useState("");
    const [payments, setPayments] = useState([]);
    const [added, setAdded] = useState(false);
    const [deleted, setDeleted] = useState(false);
    const [updated, setUpdated] = useState(false);
    const [open, setOpen] = useState(false);

    const options = {
        labels: {
            confirmable: "Confirm",
            cancellable: "Cancel"
        }
    }

    function handleOpen() {
        setOpen(!open);
    }

    const handleChangePaymentCategory = (selectedPaymentCategory) => {
        setselectedPaymentCategory(selectedPaymentCategory);
    };

    const retrievePayments = async () => {
        const response = await api.get("/payment");
        return response.data;
    }

    const retrievePaymentCategories = async () => {
        const response = await api.get("/paymentCategory");
        return response.data;
    }

    async function addPaymentHandler(payment) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = { ...payment }
                const response = await api.post("/payment", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            toast.error("Failed to Add!")
        }
    };

    const removePaymentHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/payment/${id}`);
                const newPaymentList = payments.filter((payment) => {
                    return payment.ProductID !== id;
                });
                setPayments(newPaymentList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!")
        }
    };

    const updatePaymentHandler = async (payment) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const paymentToUpdate = {
                    Id: payment.Id,
                    Availability: payment.Availability,
                    Amount: payment.Amount,
                    PaymentCategoryId: selectedPaymentCategory ? selectedPaymentCategory.value.Id : payment.ProductCategoryId,
                    PaymentCategory: selectedPaymentCategory ?
                        { Id: selectedPaymentCategory.value.Id, Name: selectedPaymentCategory.value.Name, Description: selectedPaymentCategory.value.Description }
                        :
                        { Id: payment.ProductCategoryId, Name: "", Description: "" }
                };
                await api.put("/payment", paymentToUpdate);
                setPayments(
                    payments.map((existingPayment) => {
                        return existingPayment.Id === paymentToUpdate.Id
                            ? { ...paymentToUpdate }
                            : existingPayment;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onAmountUpdate = (payment, event) => {
        const { value } = event.target;
        const data = [...rows];
        payment.Amount = value;
        initRow(data);
    };

    const onAvailabilityUpdate = (payment, event) => {
        const { value } = event.target;
        const data = [...rows];
        payment.Availability = value;
        initRow(data);
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllPayments = async () => {
            const allPayments = await retrievePayments();
            if (allPayments) setPayments(allPayments);
        };

        const getAllPaymentCategories = async () => {
            const allPaymentCategories = await retrievePaymentCategories();

            if (allPaymentCategories) setPaymentCategories(
                allPaymentCategories.map((PaymentCategoryNavigation) => {
                    return {
                        label: PaymentCategoryNavigation.Name,
                        value: PaymentCategoryNavigation
                    }
                })
            );

        };


        getAllPayments();
        getAllPaymentCategories();
        setDeleted(false);
        setUpdated(false);
        setAdded(false);


    }, [added, deleted, updated]);

    const DisplayData = payments.map(
        (payment) => {
            return (
                <tr key={payment.Id}>
                    <td>
                        {payment.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            value={payment.Amount}
                            onChange={(event) => onAmountUpdate(payment, event)}
                            name="amount"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={payment.Availability}
                            onChange={(event) => onAvailabilityUpdate(payment, event)}
                            name="availability"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={payment.PaymentCategory.Name}
                            name="PaymentCategoryID"
                            className="form-control"
                        />
                        <Select
                            value={selectedPaymentCategory}
                            onChange={handleChangePaymentCategory}
                            options={PaymentCategories}
                        />
                    </td>
                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updatePaymentHandler(payment)}
                        >
                            Update
                        </button>
                        <span> </span>
                        <button
                            className="buttonDelete"
                            onClick={() => removePaymentHandler(payment.Id)}
                        >
                            Delete
                        </button>
                    </td>
                </tr>
            )
        }
    )
    return (
        <div>
            <AddPaymentModal addPaymentHandler={addPaymentHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Amount</th>
                        <th>Availability</th>
                        <th>Category</th>
                        <th>Actions</th>
                    </tr>
                </thead>
                <tbody>
                    {DisplayData}
                </tbody>
            </table>
        </div>
    )
}

export default PaymentDataDisplay;