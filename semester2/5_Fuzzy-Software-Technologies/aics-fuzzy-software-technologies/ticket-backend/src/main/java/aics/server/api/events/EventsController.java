package aics.server.api.events;


import aics.domain.user.RoleEnum;
import aics.infrastructure.errors.TicketException;
import aics.server.api.admin.events.dtos.FetchEventDetailsResponseDto;
import aics.server.api.api_shared.ApiConstants;
import aics.server.api.events.dtos.*;
import io.quarkus.logging.Log;
import jakarta.annotation.security.PermitAll;
import jakarta.annotation.security.RolesAllowed;
import jakarta.inject.Inject;
import jakarta.ws.rs.GET;
import jakarta.ws.rs.POST;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;
import org.jboss.resteasy.reactive.RestPath;
import org.jboss.resteasy.reactive.RestResponse;


@Path(ApiConstants.API_PATH + "/events")
@PermitAll
public class EventsController {
    @Inject
    EventsActions eventsActions;

    @Path("/list-filtered")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RestResponse<FetchEventsFilteredResponseDto> handleFetchEventsFiltered(FetchEventsFilteredRequestDto fetchEventsFilteredRequestDto) {
        Log.info("Start EventsController.handleFetchEventsFiltered");
        try {
            FetchEventsFilteredResponseDto fetchEventsFilteredResponseDto = this.eventsActions.doFetchEventsFiltered(fetchEventsFilteredRequestDto);
            Log.info("End EventsController.handleFetchEventsFiltered");
            return RestResponse.ok(fetchEventsFilteredResponseDto);
        } catch (TicketException e) {
            Log.error("End EventsController.handleFetchEventsFiltered with error", e);
            return RestResponse.status(e.getStatus(), null);
        } catch (Exception e) {
            Log.error("End EventsController.handleFetchEventsFiltered with error", e);
            return RestResponse.status(RestResponse.Status.INTERNAL_SERVER_ERROR, null);
        }
    }

    @Path("/filter-options")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RestResponse<FetchEventsFilterOptionsDto> handleFetchEventsFilterOptions() {
        Log.info("Start EventsController.handleFetchEventsFilterOptions");
        try {
            FetchEventsFilterOptionsDto fetchEventsFilterOptionsDto = this.eventsActions.doFetchEventsFilterOptions();
            Log.info("End EventsController.handleFetchEventsFilterOptions");
            return RestResponse.ok(fetchEventsFilterOptionsDto);
        } catch (TicketException e) {
            Log.error("End EventsController.handleFetchEventsFilterOptions with error", e);
            return RestResponse.status(e.getStatus(), null);
        } catch (Exception e) {
            Log.error("End EventsController.handleFetchEventsFilterOptions with error", e);
            return RestResponse.status(RestResponse.Status.INTERNAL_SERVER_ERROR, null);
        }
    }

    @Path("/details/{id}")
    @GET
    @Produces(MediaType.APPLICATION_JSON)
    public RestResponse<FetchEventDetailsResponseDto> handleFetchEventDetails(@RestPath Long id) {
        Log.info("Start EventsController.handleFetchEventDetails");
        try {
            FetchEventDetailsResponseDto fetchEventDetailsResponseDto = this.eventsActions.doFetchEventDetails(id);
            Log.info("End EventsController.handleFetchEventDetails");
            return RestResponse.ok(fetchEventDetailsResponseDto);
        } catch (TicketException e) {
            Log.error("End EventsController.handleFetchEventDetails with error", e);
            return RestResponse.status(e.getStatus(), null);
        } catch (Exception e) {
            Log.error("End EventsController.handleFetchEventDetails with error", e);
            return RestResponse.status(RestResponse.Status.INTERNAL_SERVER_ERROR, null);
        }
    }

    @Path("/book-ticket")
    @POST
    @Produces(MediaType.APPLICATION_JSON)
    @RolesAllowed({RoleEnum.Values.TICKET_USER, RoleEnum.Values.TICKET_ADMIN})
    public RestResponse<BookTicketResponseDto> handleBookTicket(BookTicketRequestDto bookTicketRequestDto) {
        Log.info("Start EventsController.handleBookTicket");
        try {
            BookTicketResponseDto bookTicketResponseDto = this.eventsActions.doBookTicket(bookTicketRequestDto);
            Log.info("End EventsController.handleBookTicket");
            return RestResponse.ok(bookTicketResponseDto);
        } catch (TicketException e) {
            Log.error("End EventsController.handleBookTicket with error", e);
            return RestResponse.status(e.getStatus(), null);
        } catch (Exception e) {
            Log.error("End EventsController.handleBookTicket with error", e);
            return RestResponse.status(RestResponse.Status.INTERNAL_SERVER_ERROR, null);
        }
    }
}