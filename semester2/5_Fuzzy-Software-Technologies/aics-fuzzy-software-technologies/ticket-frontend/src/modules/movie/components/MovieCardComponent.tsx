import Card from '@mui/material/Card';
import CardActions from '@mui/material/CardActions';
import CardContent from '@mui/material/CardContent';
import CardMedia from '@mui/material/CardMedia';
import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';
import { Link } from 'react-router-dom'
import { MovieListItemDto } from '../dtos/movie-list-item-dto';
import { Avatar, Badge, CardHeader, Grid, Tooltip } from '@mui/material';
import StarIcon from '@mui/icons-material/Star';
import InsightsIcon from '@mui/icons-material/Insights';
import MoreTimeIcon from '@mui/icons-material/MoreTime';
import CalendarMonthIcon from '@mui/icons-material/CalendarMonth';
import { green, grey, red } from '@mui/material/colors';

export interface MovieCardComponentProps {
    movie: MovieListItemDto
    fuzzySearch: boolean;
    index?: number;
}

//Probably have to use arrays in order to load them dynamically
const MovieCardComponent = ({ movie, fuzzySearch, index }: MovieCardComponentProps) => {
    return (
        <Card sx={{ ":hover": { transform: 'scale(1.05)', boxShadow: 3 }, transition: 'ease', borderRadius: 5, maxWidth: 500 }}>
            {fuzzySearch && index != null && (
                <CardHeader avatar={
                    <Avatar sx={{ bgcolor: index < 3 ? green[500] : grey[500] }} aria-label="recipe">
                        {index + 1}
                    </Avatar>
                }
                    title={`Recommendation Score: ${movie.topsisScore ? movie.topsisScore.toFixed(2) : ''}`}
                />
            )}

            <CardMedia
                component="img"
                height="200"
                src={movie.imageMimePrefix + ',' + movie.image}
            />
            <CardContent>
                <Typography fontWeight={'bold'} gutterBottom variant="h5" component="div">
                    {movie.name}
                </Typography>
                <Typography variant="body2" color="text.secondary">
                    {movie.description}
                </Typography>
            </CardContent>
            <CardActions>
                <Grid container direction="row" padding={2}
                    justifyContent="space-between"
                    alignItems="center">
                    <Grid item>
                        <Button component={Link} to={'/events/list?movieId=' + movie.movieId} variant='contained' sx={{ ":hover": { backgroundColor: 'secondary' }, borderRadius: 20, backgroundColor: 'primary' }} size="small">ΠΡΟΒΟΛΕΣ</Button>
                        <Button component={Link} to={'/movies/details?movieId=' + movie.movieId} variant='outlined' sx={{ ":hover": { borderColor: '#920b17', color: '#920b17' }, color: '#E63946', backgroundColor: 'white', borderColor: '#E63946', borderRadius: 20, marginLeft: 1 }} size="small">ΠΛΗΡΟΦΟΡΙΕΣ</Button>
                    </Grid>
                    <Grid item>
                        <Tooltip title="Rating">
                            <Badge badgeContent={movie.rating} color="primary" sx={{ marginRight: 0.5 }}>
                                <StarIcon color="action" />
                            </Badge>
                        </Tooltip>
                        <Tooltip title="Duration">
                            <Badge badgeContent={movie.duration} color="primary" max={1000} sx={{ marginRight: 0.5 }}>
                                <MoreTimeIcon color="action" />
                            </Badge>
                        </Tooltip>
                        <Tooltip title="Popularity Index">
                            <Badge badgeContent={movie.popularity} color="primary" max={1000} sx={{ marginRight: 1 }}>
                                <InsightsIcon color="action" />
                            </Badge>
                        </Tooltip>
                        <Tooltip title="Year">
                            <Badge badgeContent={movie.year} color="primary" max={2100}>
                                <CalendarMonthIcon color="action" />
                            </Badge>
                        </Tooltip>
                    </Grid>
                </Grid>
            </CardActions>
        </Card >
    )
}

export default MovieCardComponent