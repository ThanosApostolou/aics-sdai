import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import "../../App.css";
import AddReviewModal from "./AddReviewModal";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";
import Select from 'react-select';

function ReviewDataDisplay() {
    const [Products, setProducts] = useState([]);
    const [selectedProduct, setSelectedProduct] = useState("");
    const [Customers, setCustomers] = useState([]);
    const [selectedCustomer, setSelectedCustomer] = useState("");
    const [reviews, setReviews] = useState([]);
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

    const handleChangeProduct = (selectedProduct) => {
        setSelectedProduct(selectedProduct);
    };

    const handleChangeCustomer = (selectedCustomer) => {
        setSelectedCustomer(selectedCustomer);
    };

    const retrieveReviews = async () => {
        const response = await api.get("/review");
        console.log(response.data)
        return response.data;
    }

    const retrieveProducts = async () => {
        const response = await api.get("/product");
        return response.data;
    }

    const retrieveCustomers = async () => {
        const response = await api.get("/eshopUser");
        return response.data;
    }

    async function addReviewHandler(review) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = { ...review }
                const response = await api.post("/review", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            console.log(e);
            toast.error("Failed to Add!")
        }
    };

    const removeReviewHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/review/${id}`);
                const newReviewList = reviews.filter((review) => {
                    return review.ReviewID !== id;
                });
                setReviews(newReviewList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!")
        }
    };

    const updateReviewHandler = async (review) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const reviewToUpdate = {
                    Id: review.Id,
                    Customer: selectedCustomer ? selectedCustomer.value.Id : review.Customer,
                    Product: selectedProduct ? selectedProduct.value.Id : review.Product,
                    Rating: review.Rating,
                    Description: review.Description,
                    ProductNavigation: selectedProduct ?
                        { Id: selectedProduct.value.Id, Name: selectedProduct.value.Name, Description: selectedProduct.value.Description,
                        Image: selectedProduct.value.Image, Availability: selectedProduct.value.Availability,
                        ProductCategory:{Id:selectedProduct.value.Id, Name : selectedProduct.value.Name, Description:selectedProduct.value.Description}  }
                        :
                        { Id: review.Product, Name: "", Description: "", Image: "", Availability: false, 
                        ProductCategory: {Id:0, Name: "", Description: "",} },
                    CustomerNavigation: selectedCustomer ?
                        { Id: selectedCustomer.value.Id, Username: selectedCustomer.value.Username, Email: selectedCustomer.value.Email, Address: selectedCustomer.value.Address }
                        :
                        { Id: review.Customer, Username: "", Email: "", Address: "" }
                };
                console.log(reviewToUpdate);
                await api.put("/review", reviewToUpdate);
                setCustomers(
                    reviews.map((existingReview) => {
                        return existingReview.Id === reviewToUpdate.Id
                            ? { ...reviewToUpdate }
                            : existingReview;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            toast.error("Failed to update!");
        }
    };

    const onRatingUpdate = (review, event) => {
        const { value } = event.target;
        const data = [...rows];
        review.Rating = value;
        initRow(data);
    };

    const onDescriptionUpdate = (review, event) => {
        const { value } = event.target;
        const data = [...rows];
        review.Description = value;
        initRow(data);
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllReviews = async () => {
            const allReviews = await retrieveReviews();
            if (allReviews) setReviews(allReviews);
        };

        const getAllProducts = async () => {
            const allProducts = await retrieveProducts();

            if (allProducts) setProducts(
                allProducts.map((Product) => {
                    return {
                        label: Product.Name,
                        value: Product
                    }
                })
            );

        };

        const getAllUsers = async () => {
            const allUsers = await retrieveCustomers();

            if (allUsers) setCustomers(
                allUsers.map((User) => {
                    return {
                        label: User.Username,
                        value: User
                    }
                })
            );
        };

        getAllReviews();
        getAllProducts();
        getAllUsers();
        setDeleted(false);
        setUpdated(false);
        setAdded(false);


    }, [added, deleted, updated]);

    const DisplayData = reviews.map(
        (review) => {
            return (
                <tr key={review.Id}>
                    <td>
                        {review.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={review.ProductNavigation.Name}
                            name="productID"
                            className="form-control"
                        />
                        <Select
                            value={selectedProduct}
                            onChange={handleChangeProduct}
                            options={Products}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={review.CustomerNavigation.Username}
                            name="customerID"
                            className="form-control"
                        />
                        <Select
                            value={selectedCustomer}
                            onChange={handleChangeCustomer}
                            options={Customers}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={review.Rating}
                            onChange={(event) => onRatingUpdate(review, event)}
                            name="rating"
                            className="form-control"
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            value={review.Description}
                            onChange={(event) => onDescriptionUpdate(review, event)}
                            name="description"
                            className="form-control"
                        />
                    </td>

                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateReviewHandler(review)}
                        >
                            Update
                        </button>
                        <span> </span>
                        <button
                            className="buttonDelete"
                            onClick={() => removeReviewHandler(review.Id)}
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
            <AddReviewModal addReviewHandler={addReviewHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Product</th>
                        <th>User</th>
                        <th>Rating</th>
                        <th>Description</th>
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

export default ReviewDataDisplay;