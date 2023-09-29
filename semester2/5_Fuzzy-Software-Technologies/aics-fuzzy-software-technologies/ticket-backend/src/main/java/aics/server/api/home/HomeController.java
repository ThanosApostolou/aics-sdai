package aics.server.api.home;

import aics.infrastructure.errors.TicketException;
import aics.server.api.api_shared.ApiConstants;
import aics.server.api.home.dtos.FetchMoviesPlayingNowResponseDto;
import io.quarkus.logging.Log;
import jakarta.annotation.security.PermitAll;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.jboss.resteasy.reactive.RestResponse;

@Path(ApiConstants.API_PATH + "/home")
@PermitAll
public class HomeController {
    @Inject
    HomeActions homeActions;

    @Path("/movies-playing-now")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RestResponse<FetchMoviesPlayingNowResponseDto> handleFetchMoviesPlayingNow() {
        Log.info("Start HomeController.handleFetchMoviesPlayingNow");
        try {
            FetchMoviesPlayingNowResponseDto fetchMoviesPlayingNowResponseDto = this.homeActions.doFetchMoviesPlayingNow();
            Log.info("End HomeController.handleFetchMoviesPlayingNow");
            return RestResponse.ok(fetchMoviesPlayingNowResponseDto);
        } catch (TicketException e) {
            Log.error("End HomeController.handleFetchMoviesPlayingNow with error", e);
            return RestResponse.status(e.getStatus(), null);
        } catch (Exception e) {
            Log.error("End HomeController.handleFetchMoviesPlayingNow with error", e);
            return RestResponse.status(RestResponse.Status.INTERNAL_SERVER_ERROR, null);
        }
    }

}