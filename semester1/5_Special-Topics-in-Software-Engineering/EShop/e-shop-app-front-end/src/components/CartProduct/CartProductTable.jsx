import React, { useState, useEffect } from 'react';
import api from "../../api/Eshop";
import "../../App.css";
import AddCartProductModal from "./AddCartProductModal";
import { confirm } from "react-confirm-box";
import toast from "react-hot-toast";
import Select from 'react-select';

function CartProductDataDisplay() {
    const [Carts, setCarts] = useState([]);
    const [selectedCart, setSelectedCart] = useState("");
    const [Products, setProducts] = useState([]);
    const [selectedProduct, setSelectedProduct] = useState("");
    const [cartProducts, setCartProducts] = useState([]);
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

    const handleChangeCart = (selectedCart) => {
        setSelectedCart(selectedCart);
    };

    const handleChangeProduct = (selectedProduct) => {
        setSelectedProduct(selectedProduct);
    };

    const retrieveCartProducts = async () => {
        const response = await api.get("/cartProduct");
        return response.data;
    }

    const retrieveCarts = async () => {
        const response = await api.get("/cart");
        return response.data;
    }

    const retrieveProducts = async () => {
        const response = await api.get("/product");
        return response.data;
    }

    async function addCartProductHandler(cartProduct) {
        try {
            handleOpen();
            const result = await confirm("Are you sure?", options);
            if (result) {
                const request = { ...cartProduct }
                const response = await api.post("/cartProduct", request)
                setAdded(true);
                toast.success("Successfully Added!")
            }
            handleOpen();
        } catch (e) {
            console.log(cartProduct)
            toast.error("Failed to Add!")
        }
    };

    const removeCartProductHandler = async (id) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                await api.delete(`/cartProduct/${id}`);
                const newCartProductList = cartProducts.filter((cartProduct) => {
                    return cartProduct.CartProductID !== id;
                });
                setCartProducts(newCartProductList);
                setDeleted(true);
                toast.success("Successfully Deleted!")
            }
        } catch (e) {
            toast.error("Failed to Delete!")
        }
    };

    const updateCartProductHandler = async (cartProduct) => {
        try {
            const result = await confirm("Are you sure?", options);
            if (result) {
                const cartProductToUpdate = {
                    Id: cartProduct.Id,
                    Cart: selectedCart ? selectedCart.value.Id : cartProduct.Cart,
                    Product: selectedProduct ? selectedProduct.value.Id : cartProduct.Product,
                    CartNavigation: selectedCart ?
                        { Id: selectedCart.value.Id, Quantity: selectedCart.value.Quantity, Customer: selectedCart.value.Customer,
                            CustomerNavigation: {Id: selectedCart.value.Customer, Username: "", Email: "", Address: ""} }
                        :
                        { Id: cartProduct.value.Cart, Quantity: cartProduct.value.Quantity, Customer: cartProduct.value.Customer,
                            CustomerNavigation: {Id: cartProduct.value.Customer, Username: "", Email: "", Address: ""} },
                        ProductNavigation: selectedProduct ?
                        { Id: selectedProduct.value.Id, Name: selectedProduct.value.Name, Description: selectedProduct.value.Description,
                        Image: selectedProduct.value.Image, Availability: selectedProduct.value.Availability,
                        ProductCategory:{Id:selectedProduct.value.Id, Name : selectedProduct.value.Name, Description:selectedProduct.value.Description}  }
                        :
                        { Id: cartProduct.Product, Name: "", Description: "", Image: "", Availability: false, 
                        ProductCategory: {Id:0, Name: "", Description: "",} }
                };
                console.log(cartProductToUpdate);
                await api.put("/cartProduct", cartProductToUpdate);
                setCartProducts(
                    cartProducts.map((existingCartProduct) => {
                        return existingCartProduct.Id === cartProductToUpdate.Id
                            ? { ...cartProductToUpdate }
                            : existingCartProduct;
                    })
                );
                setUpdated(true);
                toast.success("Successfully updated!");
            }
        } catch (e) {
            console.log(e);
            toast.error("Failed to update!");
        }
    };

    const [rows, initRow] = useState([]);

    useEffect(() => {
        const getAllCartProducts = async () => {
            const getAllCartProducts = await retrieveCartProducts();
            if (getAllCartProducts) setCartProducts(getAllCartProducts);
        };

        const getAllCarts = async () => {
            const allCarts = await retrieveCarts();

            if (allCarts) setCarts(
                allCarts.map((Cart) => {
                    return {
                        label: Cart.Id,
                        value: Cart
                    }
                })
            );

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

        getAllCartProducts();
        getAllCarts();
        getAllProducts();
        setDeleted(false);
        setUpdated(false);
        setAdded(false);


    }, [added, deleted, updated]);

    const DisplayData = cartProducts.map(
        (cartProduct) => {
            return (
                <tr key={cartProduct.Id}>
                    <td>
                        {cartProduct.Id}
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={cartProduct.CartNavigation.Id}
                            name="cartProductID"
                            className="form-control"
                        />
                        <Select
                            value={selectedCart}
                            onChange={handleChangeCart}
                            options={Carts}
                        />
                    </td>
                    <td>
                        <input
                            type="text"
                            disabled={true}
                            value={cartProduct.ProductNavigation.Name}
                            name="AdminUserId"
                            className="form-control"
                        />
                        <Select
                            value={selectedProduct}
                            onChange={handleChangeProduct}
                            options={Products}
                        />
                    </td>

                    <td>
                        <button
                            className="buttonUpdate"
                            onClick={(event) => updateCartProductHandler(cartProduct)}
                        >
                            Update
                        </button>
                        <span> </span>
                        <button
                            className="buttonDelete"
                            onClick={() => removeCartProductHandler(cartProduct.Id)}
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
            <AddCartProductModal addCartProductHandler={addCartProductHandler} open={open} handleOpen={handleOpen} />
            <table>
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Cart</th>
                        <th>Product</th>
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

export default CartProductDataDisplay;