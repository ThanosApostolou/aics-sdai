package aics.server.api.general;

import aics.infrastructure.errors.TicketException;
import aics.server.api.api_shared.ApiConstants;
import aics.server.api.general.dtos.FetchLoggedUserDetailsDto;
import io.quarkus.logging.Log;
import io.quarkus.security.Authenticated;
import jakarta.annotation.security.PermitAll;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.jboss.resteasy.reactive.RestPath;
import org.jboss.resteasy.reactive.RestResponse;

@Path(ApiConstants.API_PATH + "/general")
@PermitAll
public class GeneralController {
    @Inject
    GeneralActions generalActions;


    @Path("/logged-user-details")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @Authenticated
    public RestResponse<FetchLoggedUserDetailsDto> handleFetchLoggedUserDetails(@RestPath Long id) {
        Log.info("Start GeneralController.handleFetchLoggedUserDetails");
        try {
            FetchLoggedUserDetailsDto fetchLoggedUserDetailsDto = this.generalActions.doFetchLoggedUserDetails();
            Log.info("End GeneralController.handleFetchLoggedUserDetails");
            return RestResponse.ok(fetchLoggedUserDetailsDto);
        } catch (TicketException e) {
            Log.error("End GeneralController.handleFetchLoggedUserDetails with error", e);
            return RestResponse.status(e.getStatus(), null);
        } catch (Exception e) {
            Log.error("End GeneralController.handleFetchLoggedUserDetails with error", e);
            return RestResponse.status(RestResponse.Status.INTERNAL_SERVER_ERROR, null);
        }
    }
}