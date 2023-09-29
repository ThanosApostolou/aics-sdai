package aics.server.api.account;

import aics.domain.user.RoleEnum;
import aics.infrastructure.errors.TicketException;
import aics.server.api.account.dtos.FetchUserEventsResponseDto;
import aics.server.api.api_shared.ApiConstants;
import io.quarkus.logging.Log;
import jakarta.annotation.security.PermitAll;
import jakarta.annotation.security.RolesAllowed;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.jboss.resteasy.reactive.RestPath;
import org.jboss.resteasy.reactive.RestResponse;

@Path(ApiConstants.API_PATH + "/account")
@PermitAll
public class AccountController {
    @Inject
    AccountActions accountActions;

    @Path("/user-events")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    @RolesAllowed({RoleEnum.Values.TICKET_USER, RoleEnum.Values.TICKET_ADMIN})
    public RestResponse<FetchUserEventsResponseDto> handleFetchUserEvents(@RestPath Long id) {
        Log.info("Start AccountController.handleFetchUserEvents");
        try {
            FetchUserEventsResponseDto fetchUserEventsResponseDto = this.accountActions.doFetchUserEvents();
            Log.info("End AccountController.handleFetchUserEvents");
            return RestResponse.ok(fetchUserEventsResponseDto);
        } catch (TicketException e) {
            Log.error("End AccountController.handleFetchUserEvents with error", e);
            return RestResponse.status(e.getStatus(), null);
        } catch (Exception e) {
            Log.error("End AccountController.handleFetchUserEvents with error", e);
            return RestResponse.status(RestResponse.Status.INTERNAL_SERVER_ERROR, null);
        }
    }
}