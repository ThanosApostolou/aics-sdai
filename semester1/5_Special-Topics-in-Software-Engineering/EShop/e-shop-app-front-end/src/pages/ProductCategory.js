import ProductCategoryDataDisplay from '../components/ProductCategory/ProductCategoryTable'
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function ProductCategory() {
  return (
    <div className="container">
      <h2>Our Product Categories!!!</h2>
      <span><br /><br /></span>
      <Toaster/>
      <ProductCategoryDataDisplay />
      <div className="nav-bar-container-light">
        <BottomBar />
      </div>
    </div>
  );
}
export default ProductCategory;