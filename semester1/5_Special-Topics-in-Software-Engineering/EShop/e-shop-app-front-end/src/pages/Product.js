import ProductDataDisplay from '../components/Product/ProductTable'
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function Product() {
  return (
    <div className="container">
      <h2>Our Products!!!</h2>
      <span><br /><br /></span>
      <Toaster/>
      <ProductDataDisplay />
      <div className="nav-bar-container-light">
        <BottomBar />
      </div>
    </div>
  );
}
export default Product;