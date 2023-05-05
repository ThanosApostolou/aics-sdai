import RoleDataDisplay from '../components/Role/RoleTable'
import BottomBar from "../commons/BottomBar"
import { Toaster } from "react-hot-toast";

function Role() {
  return (
    <div className="container">
      <h2>Our Roles!!!</h2>
      <span><br /><br /></span>
      <Toaster/>
      <RoleDataDisplay />
      <div className="nav-bar-container-light">
        <BottomBar />
      </div>
    </div>
  );
}
export default Role;